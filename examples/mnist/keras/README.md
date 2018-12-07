# MNIST using Keras

Original Source: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

This is the MNIST Multi Layer Perceptron example from the [Keras examples](https://github.com/fchollet/keras/blob/master/examples), adapted for the `tf.estimator` API and TensorFlowOnSpark.

Notes:
- This example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.
- InputMode.SPARK only supports feeding data from a single RDD, so the validation dataset/code is disabled in the corresponding example.

#### Launch the Spark Standalone cluster

    export MASTER=spark://$(hostname):7077
    export SPARK_WORKER_INSTANCES=3
    export CORES_PER_WORKER=1
    export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
    export TFoS_HOME=<path to TensorFlowOnSpark>

    ${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

#### Run MNIST MLP using InputMode.TENSORFLOW

In this mode, each worker will load the entire MNIST dataset into memory (automatically downloading the dataset if needed).

    # remove any old artifacts
    rm -rf ${TFoS_HOME}/mnist_model

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp_estimator.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --input_mode tf \
    --model_dir ${TFoS_HOME}/mnist_model \
    --epochs 5 \
    --tensorboard

#### Run MNIST MLP using InputMode.SPARK

In this mode, Spark will distribute the MNIST dataset (as CSV) across the workers, so each of the two workers will see roughly half of the dataset per epoch.  Also note that InputMode.SPARK currently only supports a single input RDD, so the validation/test data is not used.

    # Convert the MNIST zip files into CSV (if not already done)
    cd ${TFoS_HOME}
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    ${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
    --output ${TFoS_HOME}/mnist/csv \
    --format csv

    # confirm that data was generated
    ls -lR ${TFoS_HOME}/mnist/csv

    # remove any old artifacts
    rm -rf ${TFoS_HOME}/mnist_model

    # train
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp_estimator.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --input_mode spark \
    --images ${TFoS_HOME}/mnist/csv/train/images \
    --labels ${TFoS_HOME}/mnist/csv/train/labels \
    --epochs 5 \
    --model_dir ${TFoS_HOME}/mnist_model \
    --tensorboard

#### Inference via saved_model_cli

The training code will automatically export a TensorFlow SavedModel, which can be used with the `saved_model_cli` from the command line, as follows:

    # path to the SavedModel export
    export SAVED_MODEL=${TFoS_HOME}/mnist_model/export/serving/*

    # use a CSV formatted test example
    IMG=$(head -n 1 $TFoS_HOME/examples/mnist/csv/test/images/part-00000)

    # introspect model
    saved_model_cli show --dir $SAVED_MODEL --all

    # inference via saved_model_cli
    saved_model_cli run --dir $SAVED_MODEL --tag_set serve --signature_def serving_default --input_exp "dense_input=[[$IMG]]"

#### Inference via TF-Serving

For online inferencing use cases, you can serve the SavedModel via a TensorFlow Serving instance as follows.  Note that TF-Serving provides both GRPC and REST APIs, but we will only
demonstrate the use of the REST API.  Also, [per the TensorFlow Serving instructions](https://www.tensorflow.org/serving/), we will run the serving instance inside a Docker container.

    # Start the TF-Serving instance in a docker container
    docker pull tensorflow/serving
    docker run -t --rm -p 8501:8501 -v "${TFoS_HOME}/mnist_model/export/serving:/models/mnist" -e MODEL_NAME=mnist tensorflow/serving &

    # GET model status
    curl http://localhost:8501/v1/models/mnist

    # GET model metadata
    curl http://localhost:8501/v1/models/mnist/metadata

    # POST example for inferencing
    curl -v -d "{\"instances\": [ {\"dense_input\": [$IMG] } ]}" -X POST http://localhost:8501/v1/models/mnist:predict

    # Stop the TF-Serving container
    docker stop $(docker ps -q)

#### Run Parallel Inferencing via Spark

For batch inferencing use cases, you can use Spark to run multiple single-node TensorFlow instances in parallel (on the Spark executors).  Each executor/instance will operate independently on a shard of the dataset.  Note that this requires that the model fits in the memory of each executor.

    # remove any old artifacts
    rm -Rf ${TFoS_HOME}/predictions

    # inference
    ${SPARK_HOME}/bin/spark-submit \
    --master $MASTER ${TFoS_HOME}/examples/mnist/keras/mnist_inference.py \
    --cluster_size 3 \
    --images_labels ${TFoS_HOME}/mnist/tfr/test \
    --export ${TFoS_HOME}/mnist_model/export/serving/* \
    --output ${TFoS_HOME}/predictions

#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

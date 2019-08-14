# MNIST using Estimator

Original Source: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_estimator

This is the [Multi-worker Training with Estimator](https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_estimator) example, adapted for TensorFlowOnSpark.

Note: this example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.

#### Launch the Spark Standalone cluster

    export MASTER=spark://$(hostname):7077
    export SPARK_WORKER_INSTANCES=3
    export CORES_PER_WORKER=1
    export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
    export TFoS_HOME=<path/to/TensorFlowOnSpark>

    ${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

#### Run using InputMode.TENSORFLOW

In this mode, each worker will load the entire MNIST dataset into memory (automatically downloading the dataset if needed).

    # remove any old artifacts
    rm -rf ${TFoS_HOME}/mnist_model
    rm -rf ${TFoS_HOME}/mnist_export

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    ${TFoS_HOME}/examples/mnist/estimator/mnist_tf.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --model_dir ${TFoS_HOME}/mnist_model \
    --export_dir ${TFoS_HOME}/mnist_export

#### Run using InputMode.SPARK

In this mode, Spark will distribute the MNIST dataset (as CSV) across the workers, so each of the workers will see only a portion of the dataset per epoch.  Also note that InputMode.SPARK currently only supports a single input RDD, so the validation/test data is not used.

    # Convert the MNIST zip files into CSV (if not already done)
    cd ${TFoS_HOME}
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --jars ${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
    ${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
    --output ${TFoS_HOME}/data/mnist

    # confirm that data was generated
    ls -lR ${TFoS_HOME}/data/mnist/csv

    # remove any old artifacts
    rm -rf ${TFoS_HOME}/mnist_model
    rm -rf ${TFoS_HOME}/mnist_export

    # train
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    ${TFoS_HOME}/examples/mnist/estimator/mnist_spark.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --images_labels ${TFoS_HOME}/data/mnist/csv/train \
    --model_dir ${TFoS_HOME}/mnist_model \
    --export_dir ${TFoS_HOME}/mnist_export

#### Inference via saved_model_cli

The training code will automatically export a TensorFlow SavedModel, which can be used with the `saved_model_cli` from the command line, as follows:

    # path to the SavedModel export
    export MODEL_BASE=${TFoS_HOME}/mnist_export
    export MODEL_VERSION=$(ls ${MODEL_BASE} | sort -n | tail -n 1)
    export SAVED_MODEL=${MODEL_BASE}/${MODEL_VERSION}

    # use a CSV formatted test example
    # converting from a flat list of 784 digits to a json array (28, 28, 1)
    cat <<EOF >reshape.py
    import sys
    import numpy as np
    vec = [int(x) for x in next(sys.stdin).split(',')]
    img = np.reshape(vec[1:], (28, 28, 1))
    print(np.array2string(img).replace('\n ', ','))
    EOF

    IMG=$(head -n 1 $TFoS_HOME/data/mnist/csv/test/part-00000 | python reshape.py)

    # introspect model
    saved_model_cli show --dir $SAVED_MODEL --all

    # inference via saved_model_cli
    saved_model_cli run --dir $SAVED_MODEL --tag_set serve --signature_def serving_default --input_exp "features=[$IMG]"

#### Inference via TF-Serving

For online inferencing use cases, you can serve the SavedModel via a TensorFlow Serving instance as follows.  Note that TF-Serving provides both GRPC and REST APIs, but we will only
demonstrate the use of the REST API.  Also, [per the TensorFlow Serving instructions](https://www.tensorflow.org/tfx/serving/docker), we will run the serving instance inside a Docker container.

    # Start the TF-Serving instance in a docker container
    docker pull tensorflow/serving
    docker run -t --rm -p 8501:8501 -v "${MODEL_BASE}:/models/mnist" -e MODEL_NAME=mnist tensorflow/serving &

    # GET model status
    curl http://localhost:8501/v1/models/mnist

    # GET model metadata
    curl http://localhost:8501/v1/models/mnist/metadata

    # POST example for inferencing
    curl -v -d "{\"instances\": [ {\"features\": $IMG } ]}" -X POST http://localhost:8501/v1/models/mnist:predict

    # Stop the TF-Serving container
    docker stop $(docker ps -q)

#### Run Parallel Inferencing via Spark

For batch inferencing use cases, you can use Spark to run multiple single-node TensorFlow instances in parallel (on the Spark executors).  Each executor/instance will operate independently on a shard of the dataset.  Note that this requires that the model fits in the memory of each executor.

    # remove any old artifacts
    rm -Rf ${TFoS_HOME}/predictions

    # inference
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    ${TFoS_HOME}/examples/mnist/estimator/mnist_inference.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --images_labels ${TFoS_HOME}/data/mnist/tfr/test \
    --export_dir ${SAVED_MODEL} \
    --output ${TFoS_HOME}/predictions


#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

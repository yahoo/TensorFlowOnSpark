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


#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

#### Inference via TF-Serving

The training code will automatically export a TensorFlow SavedModel, which can be used with TensorFlow Serving as follows.

Note: we use Docker to run the TF-Serving instance, per [recommendation](https://www.tensorflow.org/serving/).
```
# path to the SavedModel export
export MODEL=${TFoS_HOME}/mnist_model/export/serving/*

# use the CSV formatted data as a single example
IMG=$(head -n 1 $TFoS_HOME/examples/mnist/csv/test/images/part-00000)

# introspect model
saved_model_cli show --dir $MODEL --all

# inference via saved_model_cli
saved_model_cli run --dir $MODEL --tag_set serve --signature_def serving_default --input_exp "dense_input=[[$IMG]]"
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

# START the TF-Serving instance in a docker container
docker pull tensorflow/serving
docker run -t --rm -p 8501:8501 -v "${TFoS_HOME}/mnist_model/export/serving:/models/mnist" -e MODEL_NAME=mnist tensorflow/serving &

# GET model status
curl http://localhost:8501/v1/models/mnist

# GET model metadata
curl http://localhost:8501/v1/models/mnist/metadata

# POST example for inferencing
curl -v -d "{\"instances\": [ {\"dense_input\": [$IMG] } ]}" -X POST http://localhost:8501/v1/models/mnist:predict

# STOP the TF-Serving container
docker stop $(docker ps -q)
```

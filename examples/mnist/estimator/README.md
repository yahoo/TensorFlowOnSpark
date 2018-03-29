# MNIST using tf.estimator with tf.layers

Original Source: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/examples/tutorials/layers/cnn_mnist.py

This is the `tf.estimator` version of MNIST from TensorFlow's [tutorial on layers and estimators](https://www.tensorflow.org/versions/master/tutorials/layers), adapted for TensorFlowOnSpark.

Notes:
- This example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.
- To minimize code changes, this example uses InputMode.TENSORFLOW.

#### Launch the Spark Standalone cluster

    export MASTER=spark://$(hostname):7077
    export SPARK_WORKER_INSTANCES=3
    export CORES_PER_WORKER=1
    export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
    export TFoS_HOME=<path to TensorFlowOnSpark>

    ${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

#### Run MNIST using InputMode.TENSORFLOW

In this mode, each worker will load the entire MNIST dataset into memory (automatically downloading the dataset if needed).

    # remove any old artifacts
    rm -rf ${TFoS_HOME}/mnist_model

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.task.maxFailures=1 \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/estimator/mnist_estimator.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --model ${TFoS_HOME}/mnist_model

#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh


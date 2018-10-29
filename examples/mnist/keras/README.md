# MNIST using Keras

Original Source: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

This is the MNIST Multi Layer Perceptron example from the [Keras examples](https://github.com/fchollet/keras/blob/master/examples), adapted for TensorFlowOnSpark.

Notes:
- This example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.
- Keras currently saves model checkpoints as [HDF5](https://support.hdfgroup.org/HDF5/) using the [h5py package](http://www.h5py.org/).  Unfortunately, this is not currently supported on HDFS.  Consequently, this example demonstrates how to save standard TensorFlow model checkpoints on HDFS via a Keras LambdaCallback.  If you don't need HDFS support, you can use the standard ModelCheckpoint instead.
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
    rm -rf ${TFoS_HOME}/mnist_model ${TFoS_HOME}/mnist_export

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --input_mode tf \
    --model_dir ${TFoS_HOME}/mnist_model \
    --export_dir ${TFoS_HOME}/mnist_export \
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
    rm -rf ${TFoS_HOME}/mnist_model ${TFoS_HOME}/mnist_export

    # train and validate
    ${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    ${TFoS_HOME}/examples/mnist/keras/mnist_mlp.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --input_mode spark \
    --images ${TFoS_HOME}/mnist/csv/train/images \
    --labels ${TFoS_HOME}/mnist/csv/train/labels \
    --epochs 5 \
    --model_dir ${TFoS_HOME}/mnist_model \
    --export_dir ${TFoS_HOME}/mnist_export \
    --tensorboard

#### Shutdown the Spark Standalone cluster

    ${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh


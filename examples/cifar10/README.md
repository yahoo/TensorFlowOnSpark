# CIFAR-10 Multi-GPU CNN

Original Source: https://github.com/tensorflow/tensorflow/blob/eaceadc3c421bb41cfbf607ca832b3b9b2ad2507/tensorflow/g3doc/tutorials/deep_cnn/index.md

The following is the Multi-GPU CNN Tutorial, adapted for TensorFlowOnSpark. This example demonstrates how to use multiple GPU cards on a single node. Note: since YARN currently cannot allocate GPU resources directly, we currently use RAM as a proxy, so in our case, 1GPU == 27GB.  You may need to adjust this for your grid.

Please ensure that you have followed [these instructions](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN) first.

Also, you will need to download the CIFAR-10 dataset per the [original example](https://github.com/tensorflow/tensorflow/blob/eaceadc3c421bb41cfbf607ca832b3b9b2ad2507/tensorflow/g3doc/tutorials/deep_cnn/index.md).

#### Package the code as a Python zip/module

    export TFoS_HOME=<path to TensorFlowOnSpark>
    pushd ${TFoS_HOME}/examples/cifar10; zip -r ~/cifar10.zip .; popd

#### Run Multi-GPU CNN on Spark

    # set environment variables (if not already done)
    export PYTHON_ROOT=~/Python
    export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
    export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
    export PATH=${PYTHON_ROOT}/bin/:$PATH
    export QUEUE=gpu
    export CIFAR10_DATA=<HDFS path to your downloaded files>

    # for CPU mode:
    # export QUEUE=default
    # --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server" \
    # remove --driver-library-path

    # hadoop fs -rm -r cifar10_train
    export NUM_GPU=2
    export MEMORY=$((NUM_GPU * 27))
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 1 \
    --executor-memory ${MEMORY}G \
    --py-files ${TFoS_HOME}/tfspark.zip,cifar10.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/cifar10/cifar10_multi_gpu_train.py \
    --data_dir ${CIFAR10_DATA} \
    --train_dir hdfs://default/user/${USER}/cifar10_train \
    --max_steps 1000 \
    --num_gpus ${NUM_GPU}

### Run evaluation on Spark

    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 1 \
    --executor-memory 27G \
    --py-files ${TFoS_HOME}/tfspark.zip,cifar10.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="lib64:/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="lib64:/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/cifar10/cifar10_eval.py \
    --data_dir ${CIFAR10_DATA} \
    --checkpoint_dir hdfs://default/user/${USER}/cifar10_train \
    --eval_dir hdfs://default/user/${USER}/cifar10_eval \
    --run_once

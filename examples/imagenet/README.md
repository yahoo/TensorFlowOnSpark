# Inception V3 CNN

Original Source: https://github.com/tensorflow/models/tree/master/inception

In this example, we leave the code largely untouched, leveraging TensorFlowOnSpark to launch the cluster in the Hadoop grid.
To view the differences, you can compare the original `imagenet_distributed_train.py` with the version here.

These instructions are intended for a Spark/YARN grid, so please ensure that you have followed [these instructions](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN) first.

Also, you will need to [download the Imagenet dataset per the original example](https://github.com/tensorflow/models/tree/master/inception#getting-started).

#### Package the inception code as a Python zip/module

    export TFoS_HOME=<path to TensorFlowOnSpark>
    pushd ${TFoS_HOME}/examples/imagenet; zip -r ~/inception.zip inception; popd

#### Run distributed CNN on Spark

    # set environment variables (if not already done)
    export PYTHON_ROOT=~/Python
    export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
    export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
    export PATH=${PYTHON_ROOT}/bin/:$PATH
    export QUEUE=gpu
    export IMAGENET_DATA=<HDFS path to your downloaded files>

    # for CPU mode:
    # export QUEUE=default
    # --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server" \
    # remove --driver-library-path

    # hadoop fs -rm -r imagenet_train
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 27G \
    --py-files ${TFoS_HOME}/tfspark.zip,inception.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/imagenet/inception/imagenet_distributed_train.py \
    --data_dir ${IMAGENET_DATA} \
    --train_dir hdfs://default/user/${USER}/imagenet_train \
    --max_steps 1000 \
    --subset train
    # to use infiniband, replace the last line with --subset train --rdma

#### Run evaluation job on Spark

To evaluate the model, run the following job after the training has completed.  This will calculate the "precision @ 1" metric for the trained model.  Note: since we only trained for 1000 steps, the reported metric will be very poor.  So, to train a better model, you can increase the `--max_steps` above, and then run the evaluation job in parallel by removing the `--run_once` argument.  This will periodically calculate the metric while training is in progress.  You can terminate training and/or eval at any time using the standard `yarn application -kill <applicationId>` command, and the latest model will be stored in your `imagenet_train` HDFS directory.

    # for CPU mode:
    # export QUEUE=default
    # --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server" \
    # remove --driver-library-path

    # hadoop fs -rm -r imagenet_eval
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 1 \
    --executor-memory 27G \
    --py-files ${TFoS_HOME}/tfspark.zip,inception.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/imagenet/inception/imagenet_eval.py \
    --data_dir ${IMAGENET_DATA} \
    --checkpoint_dir hdfs://default/user/${USER}/imagenet_train \
    --eval_dir hdfs://default/user/${USER}/imagenet_eval \
    --subset validation \
    --run_once

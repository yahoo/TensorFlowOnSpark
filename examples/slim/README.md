# TF-Slim Inception

Original Source: https://github.com/tensorflow/models/tree/master/slim

This example demonstrates the conversion of a TF-Slim image classification application.

Please ensure that you have followed [these instructions](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN) first.
And, you will need to [download an image dataset](https://github.com/tensorflow/models/tree/master/slim) per the original instructions.

#### Package the code as a Python zip/module

    export TFoS_HOME=<Path to TensorFlowOnSpark>
    pushd ${TFoS_HOME}/examples/slim; zip -r ~/slim.zip .; popd

#### Train TF-Slim Classifier

    # set environment variables (if not already done)
    export PYTHON_ROOT=~/Python
    export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
    export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
    export PATH=${PYTHON_ROOT}/bin/:$PATH
    export QUEUE=gpu
    export DATASET_DIR=<HDFS path to your downloaded files>

    # for CPU mode:
    # export QUEUE=default
    # --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/jre/lib/amd64/server" \
    # remove --driver-library-path

    # hadoop fs -rm -r slim_train
    export NUM_GPU=1
    export MEMORY=$((NUM_GPU * 27))
    ${SPARK_HOME}/bin/spark-submit --master yarn --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 3 \
    --executor-memory ${MEMORY}G \
    --py-files ${TFoS_HOME}/tfspark.zip,slim.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.ui.view.acls=* \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/slim/train_image_classifier.py \
    --dataset_dir ${DATASET_DIR} \
    --train_dir hdfs://default/user/${USER}/slim_train \
    --dataset_name imagenet \
    --dataset_split_name train \
    --model_name inception_v3 \
    --max_number_of_steps 1000 \
    --num_gpus ${NUM_GPU} \
    --batch_size 32 \
    --num_ps_tasks 1

#### Evaluate TF-Slim Classifier

    # hadoop fs -rm -r slim_eval
    ${SPARK_HOME}/bin/spark-submit --master yarn --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 1 \
    --executor-memory 27G \
    --py-files ${TFoS_HOME}/tfspark.zip,slim.zip \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.ui.view.acls=* \
    --conf spark.task.maxFailures=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64:$JAVA_HOME/jre/lib/amd64/server" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    ${TFoS_HOME}/examples/slim/eval_image_classifier.py \
    --dataset_dir ${DATASET_DIR} \
    --dataset_name imagenet \
    --dataset_split_name validation \
    --model_name inception_v3 \
    --checkpoint_path hdfs://default/user/${USER}/slim_train \
    --eval_dir hdfs://default/user/${USER}/slim_eval

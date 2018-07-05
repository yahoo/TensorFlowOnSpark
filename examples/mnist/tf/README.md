## Running distributed MNIST training / inference

### _using Dataset_
```bash
# for CPU mode:
# export QUEUE=default
# remove references to $LIB_CUDA

# hdfs dfs -rm -r mnist_model
# hdfs dfs -rm -r predictions

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 4 \
--executor-memory 27G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist_dataset.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA:$LIB_JVM:$LIB_HDFS \
--driver-library-path=$LIB_CUDA \
TensorFlowOnSpark/examples/mnist/tf/mnist_spark_dataset.py \
${TF_ROOT}/${TF_VERSION}/examples/mnist/tf/mnist_spark_dataset.py \
--images_labels mnist/csv2/train \
--format csv2 \  
--mode train \
--model mnist_model

# to use inference mode, change `--mode train` to `--mode inference` and add `--output predictions`
# one item in csv2 format is `image | label`, to use input data in TFRecord format, change `--format csv` to `--format tfr`
# to use infiniband, add `--rdma`
```

### _using QueueRunners_
```bash
# for CPU mode:
# export QUEUE=default
# remove references to $LIB_CUDA

# hdfs dfs -rm -r mnist_model
# hdfs dfs -rm -r predictions

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 4 \
--executor-memory 27G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA:$LIB_JVM:$LIB_HDFS \
--driver-library-path=$LIB_CUDA \
TensorFlowOnSpark/examples/mnist/tf/mnist_spark.py \
--images mnist/tfr/train/images \
--labels mnist/tfr/train/labels \
--format csv \
--mode train \
--model mnist_model

# to use inference mode, change `--mode train` to `--mode inference` and add `--output predictions`
# to use input data in TFRecord format, change `--format csv` to `--format tfr`
# to use infiniband, add `--rdma`
```

### _using Spark ML Pipeline_
```bash
# for CPU mode:
# export QUEUE=default
# remove references to $LIB_CUDA

# hdfs dfs -rm -r mnist_model
# hdfs dfs -rm -r mnist_export
# hdfs dfs -rm -r tfrecords
# hdfs dfs -rm -r predictions

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 4 \
--executor-memory 27G \
--jars hdfs:///user/${USER}/tensorflow-hadoop-1.0-SNAPSHOT.jar \  
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist_pipeline.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_CUDA:$LIB_JVM:$LIB_HDFS \
--driver-library-path=$LIB_CUDA \
TensorFlowOnSpark/examples/mnist/tf/mnist_spark_pipeline.py \
--images mnist/csv/train/images \
--labels mnist/csv/train/labels \
--tfrecord_dir tfrecords \
--format csv \
--model_dir mnist_model \
--export_dir mnist_export \
--train \
--inference_mode signature \
--inference_output predictions

# to use input data in TFRecord format, change `--format csv` to `--format tfr`
# tensorflow-hadoop-1.0-SNAPSHOT.jar is needed for transforming csv input to TFRecord 
# `--tfrecord_dir` is needed for temporarily saving dataframe to TFRecord on hdfs
```

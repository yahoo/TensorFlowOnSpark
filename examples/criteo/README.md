# Learning Click-Through Rate at Scale with Tensorflow on Spark

## Introduction 
This project consists of learning a click-throughrate model at scale using TensorflowOnSpark technology.
Criteo released a 1TB dataset: http://labs.criteo.com/2013/12/download-terabyte-click-logs/
In order to promote Google cloud technology, Google published a solution to train a model at scale using there 
proprietary platform : https://cloud.google.com/blog/big-data/2017/02/using-google-cloud-machine-learning-to-predict-clicks-at-scale 

Instead, we propose a solution based on open source technology that can be leveraged on any cloud, 
or private cluster relying on spark.

We demonstrate how Tensorflow on Spark (https://github.com/yahoo/TensorFlowOnSpark) can be used to reach the state of the art when it comes to predicting the proba of click at scale.
Notice that the goal here is not to produce the best pCTR predictor, but rather establish a open method that still reaches the best performance published so far on this dataset.
Hence, our solutions remains very simple, and rely solely on basic feature extraction, cross-features and hashing, the all trained on logistic regression.

## Install and test TF on spark
Before making use of this code, please make sure you can install TF on spark on your cluster and
run the mnist example as illustrated here:
https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN
By so doing, you should make sure that did set up the following variables correctly:

```
export JAVA_HOME=
export HADOOP_HOME=
export SPARK_HOME=
export HADOOP_HDFS_HOME=
export SPARK_HOME=
export PYTHON_ROOT=./Python
export PATH=${PATH}:${HADOOP_HOME}/bin:${SPARK_HOME}/bin:${HADOOP_HDFS_HOME}/bin:${SPARK_HOME}/bin:${PYTHON_ROOT}/bin
export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=/usr/bin/python"
export QUEUE=default
export LIB_HDFS=
export LIB_JVM=
```

## Data set

The raw data can be accessed here: http://labs.criteo.com/2013/12/download-terabyte-click-logs/

### Download the data set
```
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
	curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz
	aws s3 mv  day_${i}.gz s3://criteo-display-ctr-dataset/released/
done
```

### Upload training data on your AWS s3 using Pig

```
%declare awskey yourkey
%declare awssecretkey yoursecretkey
SET mapred.output.compress 'true';
SET mapred.output.compression.codec 'org.apache.hadoop.io.compress.BZip2Codec';
train_data = load 's3n://${awskey}:${awssecretkey}@criteo-display-ctr-dataset/released/day_{[0-9],1[0-9],2[0-2]}.gz ';
train_data = FOREACH (GROUP train_data BY ROUND(10000* RANDOM()) PARALLEL 10000) GENERATE FLATTEN(train_data); 
store train_data into 's3n://${awskey}:${awssecretkey}@criteo-display-ctr-dataset/data/training/' using PigStorage();
```
We here divide the training data in 10000 chunks, which will allow TFonSpark to reduce its memory usage.

### Upload validation data on your AWS s3 using Pig
```
%declare awskey yourkey
%declare awssecretkey yoursecretkey
SET mapred.output.compress 'true';
SET mapred.output.compression.codec 'org.apache.hadoop.io.compress.BZip2Codec';
train_data = load 's3n://${awskey}:${awssecretkey}@criteo-display-ctr-dataset/released/day_23.gz';
train_data = FOREACH (GROUP train_data BY ROUND(100* RANDOM()) PARALLEL 100) GENERATE FLATTEN(train_data); 
store train_data into 's3n://${awskey}:${awssecretkey}@criteo-display-ctr-dataset/data/validation' using PigStorage();
```






## Running the example

Set up task variables 
```
export TRAINING_DATA=hdfs_path_to_training_data_directory
export VALIDATION_DATA=hdfs_path_to_validation_data_directory
export MODEL_OUTPUT=hdfs://default/tmp/criteo_ctr_prediction
```
Run command:

```
${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 12 \
--executor-memory 27G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/criteo/spark/criteo_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/${USER}/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH="$LIB_HDFS:$LIB_JVM" \
--conf spark.executorEnv.HADOOP_HDFS_HOME="$HADOOP_HDFS_HOME" \
--conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" \
TensorFlowOnSpark/examples/criteo/spark/criteo_spark.py \
--mode train \
--data ${TRAINING_DATA} \
--validation ${VALIDATION_DATA} \
--steps 1000000 \
--model ${MODEL_OUTPUT} --tensorboard \
--tensorboardlogdir ${MODEL_OUTPUT}
```
## Tensorboard tracking:

By connecting to the Web UI tracker of your application,
you be able to retrieve the tensorboard URL in the stdout of the driver, e.g.:
```
 TensorBoard running at:       http://10.4.112.234:36911
```


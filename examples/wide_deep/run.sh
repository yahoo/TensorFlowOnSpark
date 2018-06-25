#!/bin/bash

export PYTHON_ROOT=./Python
export LD_LIBRARY_PATH=${PATH}
export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
export PATH=${PYTHON_ROOT}/bin/:$PATH
export HADOOP_HOME=/opt/cloudera/parcels/CDH-5.11.0-1.cdh5.11.0.p0.34
# set paths to libjvm.so, libhdfs.so
export LIB_HDFS=$HADOOP_HOME/lib64
export LIB_HADOOP=$HADOOP_HOME/lib/hadoop/lib/native
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server

model_dir="hdfs:///user/root/tfos/model/"
## restore model if run well
hadoop dfs -rmr $model_dir/*
#hadoop dfs -mkdir $model_dir
#hadoop dfs -chmod 777 $model_dir

export_dir="hdfs:///user/root/tfos/export/"
## restore if run well 
hadoop dfs -rmr $export_dir/*
#hadoop dfs -mkdir $export_dir
#hadoop dfs -chmod 777 $export_dir

model="estimator"
cur_date=`date +%Y%m%d`

spark_submit_cmd="/usr/local/spark/bin/spark-submit \
--name tfos_$model"_"$cur_date \
--master yarn-cluster \
--queue online \
--deploy-mode cluster \
--num-executors 100 \
--executor-memory 20g \
--driver-memory 40g \
--driver-cores 2 \
--py-files d.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs:///user/root/tensorflow/Python.zip#Python \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
./d.py \
--num_ps 30 \
--task_num 70 \
--export_dir $export_dir \
--model_dir $model_dir"

echo $spark_submit_cmd
$spark_submit_cmd


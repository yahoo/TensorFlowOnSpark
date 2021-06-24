#!/bin/bash -x
#export SPARK_HOME=/opt/spark
#export SPARK_LOCAL_IP=127.0.0.1
#export PATH=$SPARK_HOME/bin:$PATH
#
## Start Spark Standalone Cluster
#export SPARK_CLASSPATH=./lib/tensorflow-hadoop-1.0-SNAPSHOT.jar
#export MASTER=spark://$(hostname):7077
#export SPARK_WORKER_INSTANCES=2; export CORES_PER_WORKER=1
#export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 1G ${MASTER}

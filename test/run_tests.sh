#!/bin/bash

if [ -z "$SPARK_HOME" ]; then
  echo "Please set SPARK_HOME environment variable"
fi

if [ -z "$TFoS_HOME" ]; then
  echo "Please set TFoS_HOME environment variable"
fi

# Start Spark Standalone Cluster
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3; export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c ${CORES_PER_WORKER} -m 3G ${MASTER}

# Run Tests
python -m unittest discover

# Stop Spark Standalone Cluster
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

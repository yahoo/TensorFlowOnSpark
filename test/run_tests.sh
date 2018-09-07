#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ -z "$SPARK_HOME" ]; then
  echo "Please set SPARK_HOME environment variable"
  exit 1
fi

if [ -z "$SPARK_CLASSPATH" ]; then
  echo "Please add the path to tensorflow-hadoop-*.jar to the SPARK_CLASSPATH environment variable"
  exit 1
fi

# Start Spark Standalone Cluster
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2; export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c ${CORES_PER_WORKER} -m 1G ${MASTER}

# Run tests
python -m unittest discover -s $DIR
EXIT_CODE=$?

# Stop Spark Standalone Cluster
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh

exit $EXIT_CODE

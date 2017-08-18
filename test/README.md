# Unit/Integration Tests

## Requirements

Since TensorFlowOnSpark (TFoS) is literally an integration of TensorFlow and Spark, these tests assume your environment has:
- Spark installed at `${SPARK_HOME}`
- Python installed with tensorflow
- TFoS installed via `pip install -e .` (for easier coding/iteration)

Note: the tests that use Spark will require a local Spark Standalone cluster (vs. Spark local mode), since TFoS assumes that the executors run in separate processes.  This is true for distributed clusters (Standalone and YARN), but not true for the non-distributed Spark local mode, since the executors are just launched as threads in a single process.

## Instructions

```
# Start Spark Standalone Cluster
export TFoS_HOME=<path_to_TFoS>
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3; export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

# Run Tests
cd ${TFoS_HOME}/test
python -m unittest discover

# Stop Spark Standalone Cluster
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

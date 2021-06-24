# Unit/Integration Tests

## Requirements

Since TensorFlowOnSpark (TFoS) is literally an integration of TensorFlow and Spark, these tests assume your environment has:
- Spark installed at `${SPARK_HOME}`
- Python installed with tensorflow
- TFoS installed via `pip install -e .` (for easier coding/iteration)

Note: the tests that use Spark will require a local Spark Standalone cluster (vs. Spark Local mode), since TFoS assumes that the executors run in separate processes.  This is true for distributed clusters (Standalone and YARN), but not true for the non-distributed Spark Local mode, since the executors are just launched as threads in a single process.

## Instructions

1. Setup ENV variables to point to your Spark and TensorFlowOnSpark.
```bash
export SPARK_HOME=<path_to_Spark>
export TFoS_HOME=<path_to_TFoS>
export PYTHONPATH=${SPARK_HOME}/python
export SPARK_CLASSPATH=${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar
```
2. Run script to automatically start Spark Standalone cluster, run all tests, and shutdown the cluster, OR
```bash
cd ${TFoS_HOME}/tests
./run_tests.sh
```
3. OPTIONAL: manually start/stop the Spark Standalone cluster (when iterating on code).
```
# Start Spark Standalone cluster
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2; export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 3G ${MASTER}

# Develop code, run tests, repeat...
cd ${TFoS_HOME}/tests
python -m unittest discover

# Stop Spark Standalone cluster when done
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

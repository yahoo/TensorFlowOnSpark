# MNIST

Original Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py

Notes:
- This assumes that you have already [installed Spark, TensorFlow, and TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_Standalone)
- This code has been heavily modified to support different input formats (CSV and TFRecords) and different data ingestion methods (`InputMode.TENSORFLOW` and `InputMode.SPARK`).

### Download MNIST data

```
mkdir ${TFoS_HOME}/mnist
pushd ${TFoS_HOME}/mnist
curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
popd
```

### Convert the MNIST zip files using Spark

```
cd ${TFoS_HOME}
# rm -rf examples/mnist/csv
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output examples/mnist/csv \
--format csv
ls -lR examples/mnist/csv
```

### Start Spark Standalone Cluster

```
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}
```

### Run distributed MNIST training using `InputMode.SPARK`

```
# rm -rf mnist_model
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model

ls -l mnist_model
```

### Run distributed MNIST inference using `InputMode.SPARK`

```
# rm -rf predictions
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/test/images \
--labels examples/mnist/csv/test/labels \
--mode inference \
--format csv \
--model mnist_model \
--output predictions

less predictions/part-00000
```

The prediction result should look like:
```
2017-02-10T23:29:17.009563 Label: 7, Prediction: 7
2017-02-10T23:29:17.009677 Label: 2, Prediction: 2
2017-02-10T23:29:17.009721 Label: 1, Prediction: 1
2017-02-10T23:29:17.009761 Label: 0, Prediction: 0
2017-02-10T23:29:17.009799 Label: 4, Prediction: 4
2017-02-10T23:29:17.009838 Label: 1, Prediction: 1
2017-02-10T23:29:17.009876 Label: 4, Prediction: 4
2017-02-10T23:29:17.009914 Label: 9, Prediction: 9
2017-02-10T23:29:17.009951 Label: 5, Prediction: 6
2017-02-10T23:29:17.009989 Label: 9, Prediction: 9
2017-02-10T23:29:17.010026 Label: 0, Prediction: 0
```

### Shutdown Spark cluster

```
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

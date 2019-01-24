# Wide & Deep Model

Original Source: https://github.com/tensorflow/models/tree/master/official/wide_deep

In this example, we use TensorFlowOnSpark, along with the [tf.estimator.train_and_evaluate](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) API, to convert a single-node TensorFlow application into a distributed one.


## How to run

For simplicity, we'll use Spark Standalone on a single node.  If you haven't already done so, you should try the [Getting Started on Spark Standalone](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_Standalone) instructions.

#### Clone this repository (if not already done)

```bash
git clone https://github.com/yahoo/TensorFlowOnSpark.git
cd TensorFlowOnSpark
export TFoS_HOME=$(pwd)
```

#### Clone the TensorFlow Models repository

This example depends on code in the [TensorFlow Models](https://github.com/tensorflow/models) repository, so you will have to clone the repo:
```bash
git clone https://github.com/tensorflow/models.git
cd models
export TF_MODELS=$(pwd)
```

#### Start Spark Standalone Cluster

```bash
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}
```

### Download the UCI Census Income Dataset

```bash
cd ${TFoS_HOME}/examples/wide_deep

python census_dataset.py
```

### Run Distributed Wide & Deep

```bash
cd ${TFoS_HOME}/examples/wide_deep

# rm -Rf /tmp/census_model; \
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files census_dataset.py,wide_deep_run_loop.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.task.maxFailures=1 \
--conf spark.stage.maxConsecutiveAttempts=1 \
census_main.py \
--cluster_size 3
```

The TensorFlow logs for each node will be available in `stderr` link of each executor in the Spark UI.  For example, in the log of the `master` node, you should see something like the following:
```
I0124 09:33:27.728477 4486518208 tf_logging.py:115] Finished evaluation at 2019-01-24-17:33:27
I0124 09:33:27.729230 4486518208 tf_logging.py:115] Saving dict for global step 1729: accuracy = 0.82875, accuracy_baseline = 0.76325, auc = 0.8827834, auc_precision_recall = 0.7127151, average_loss = 0.3687935, global_step = 1729, label/mean = 0.23675, loss = 14.7517395, precision = 0.7119741, prediction/mean = 0.261756, recall = 0.46462512
```

#### Shutdown Standalone Cluster

```bash
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

# MNIST using Estimator

Original Source: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator

This is the [Multi-worker Training with Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator) example, adapted for TensorFlowOnSpark.

Note: this example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.

#### Launch the Spark Standalone cluster
```bash
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export TFoS_HOME=<path/to/TensorFlowOnSpark>

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c $CORES_PER_WORKER -m 3G ${MASTER}
```

#### Train via InputMode.TENSORFLOW

In this mode, each worker will load the entire MNIST dataset into memory (automatically downloading the dataset if needed).
```bash
# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm -rf ${TFoS_HOME}/mnist_export

# train and validate
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/examples/mnist/estimator/mnist_tf.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--model_dir ${TFoS_HOME}/mnist_model \
--export_dir ${TFoS_HOME}/mnist_export

# confirm model
ls -lR ${TFoS_HOME}/mnist_model
ls -lR ${TFoS_HOME}/mnist_export
```

#### Train via InputMode.SPARK

In this mode, Spark will distribute the MNIST dataset (as CSV) across the workers, so each of the workers will see only a portion of the dataset per epoch.  Also note that InputMode.SPARK currently only supports a single input RDD, so the validation/test data is not used.

Convert the MNIST zip files into CSV, if not already done:
```
cd ${TFoS_HOME}
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--jars ${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output ${TFoS_HOME}/data/mnist

# confirm that data was generated
ls -lR ${TFoS_HOME}/data/mnist/csv
```

Then, train as follows:
```bash
# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm -rf ${TFoS_HOME}/mnist_export

# train
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/examples/mnist/estimator/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images_labels ${TFoS_HOME}/data/mnist/csv/train \
--model_dir ${TFoS_HOME}/mnist_model \
--export_dir ${TFoS_HOME}/mnist_export

# confirm model
ls -lR ${TFoS_HOME}/mnist_model
ls -lR ${TFoS_HOME}/mnist_export
```

#### Inference via saved_model_cli

The training code will automatically export a TensorFlow SavedModel, which can be used with the `saved_model_cli` from the command line, as follows:

```bash
# path to the SavedModel export
export MODEL_BASE=${TFoS_HOME}/mnist_export
export MODEL_VERSION=$(ls ${MODEL_BASE} | sort -n | tail -n 1)
export SAVED_MODEL=${MODEL_BASE}/${MODEL_VERSION}

# use a CSV formatted test example (reshaping from [784] to [28, 28, 1])
IMG=$(head -n 1 ${TFoS_HOME}/data/mnist/csv/test/part-00000 | python ${TFoS_HOME}/examples/utils/mnist_reshape.py)

# introspect model
saved_model_cli show --dir $SAVED_MODEL --all

# inference via saved_model_cli
saved_model_cli run --dir $SAVED_MODEL --tag_set serve --signature_def serving_default --input_exp "conv2d_input=[$IMG]"
```

#### Inference via TF-Serving

For online inferencing use cases, you can serve the SavedModel via a TensorFlow Serving instance as follows.  Note that TF-Serving provides both GRPC and REST APIs, but we will only
demonstrate the use of the REST API.  Also, [per the TensorFlow Serving instructions](https://www.tensorflow.org/tfx/serving/docker), we will run the serving instance inside a Docker container.
```bash
# path to the SavedModel export
export MODEL_BASE=${TFoS_HOME}/mnist_export
IMG=$(head -n 1 ${TFoS_HOME}/data/mnist/csv/test/part-00000 | python ${TFoS_HOME}/examples/utils/mnist_reshape.py)

# Start the TF-Serving instance in a docker container
docker pull tensorflow/serving
docker run -d --rm -p 8501:8501 -v "${MODEL_BASE}:/models/mnist" -e MODEL_NAME=mnist tensorflow/serving

# wait for server to start
sleep 2

# GET model status
curl http://localhost:8501/v1/models/mnist

# GET model metadata
curl http://localhost:8501/v1/models/mnist/metadata

# POST example for inferencing
curl -v -d "{\"instances\": [ {\"conv2d_input\": $IMG } ]}" -X POST http://localhost:8501/v1/models/mnist:predict

# Stop the TF-Serving container
docker stop $(docker ps -q)
```

#### Parallel Inferencing via Spark

For batch inferencing use cases, you can use Spark to run multiple single-node TensorFlow instances in parallel (on the Spark executors).  Each executor/instance will operate independently on a shard of the dataset.  Note that this requires that the model fits in the memory of each executor.
```bash
# path to the SavedModel export
export MODEL_BASE=${TFoS_HOME}/mnist_export
export MODEL_VERSION=$(ls ${MODEL_BASE} | sort -n | tail -n 1)
export SAVED_MODEL=${MODEL_BASE}/${MODEL_VERSION}

# remove any old artifacts
rm -Rf ${TFoS_HOME}/predictions

# inference
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/examples/mnist/estimator/mnist_inference.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images_labels ${TFoS_HOME}/data/mnist/tfr/test \
--export_dir ${SAVED_MODEL} \
--output ${TFoS_HOME}/predictions

# confirm predictions
ls -lR ${TFoS_HOME}/predictions
```

#### Train and Inference via Spark ML Pipeline API

Spark also includes an [ML Pipelines API](https://spark.apache.org/docs/latest/ml-pipeline.html), built on Spark DataFrames and intended for ML applications.  Since this API is targeted towards building ML pipelines in Spark, only InputMode.SPARK is supported for this API.  However, a `dfutil` library is provided to read simple TFRecords into a Spark DataFrame.  Note that complex TFRecords are not supported, since they cannot be easily represented in Spark DataFrames.

```bash
# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm -rf ${TFoS_HOME}/mnist_export

# train w/ CSV
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--jars ${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
${TFoS_HOME}/examples/mnist/estimator/mnist_pipeline.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images_labels ${TFoS_HOME}/data/mnist/csv/train \
--format csv \
--mode train \
--model_dir ${TFoS_HOME}/mnist_model \
--export_dir ${TFoS_HOME}/mnist_export

# train with TFRecords
# --images_labels ${TFoS_HOME}/data/mnist/tfr/train \
# --format tfr \

# confirm model
ls -lR ${TFoS_HOME}/mnist_model
ls -lR ${TFoS_HOME}/mnist_export

# inference w/ CSV using exported saved_model
export MODEL_BASE=${TFoS_HOME}/mnist_export
export MODEL_VERSION=$(ls ${MODEL_BASE} | sort -n | tail -n 1)
export SAVED_MODEL=${MODEL_BASE}/${MODEL_VERSION}

# remove any old artifacts
rm -rf ${TFoS_HOME}/predictions

# inference with CSV
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--jars ${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
${TFoS_HOME}/examples/mnist/estimator/mnist_pipeline.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images_labels ${TFoS_HOME}/data/mnist/csv/test \
--format csv \
--mode inference \
--export_dir ${SAVED_MODEL} \
--output ${TFoS_HOME}/predictions

# inference with TFRecords
# --images_labels ${TFoS_HOME}/data/mnist/tfr/test \
# --format tfr \

# confirm predictions
ls -lR ${TFoS_HOME}/predictions
```

#### Train via InputMode.SPARK with Spark Streaming

Spark also includes a streaming mode, which allows you feed data to your Spark applications in an online/streaming mode vs. reading a static list of files from disk. In this mode, Spark watches a location on disk (or listens on a network port) for new data to arrive and batches the incoming data into a sequence of RDDs for your application.

This example is essentially the same as the one above, except it replaces the Spark RDD with a Spark Streaming RDD (DStream).  Note that there is no final export to saved_model, since training is essentially ongoing.

Convert the MNIST zip files into CSV, if not already done:
```
cd ${TFoS_HOME}
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--jars ${TFoS_HOME}/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output ${TFoS_HOME}/data/mnist

# confirm that data was generated
ls -lR ${TFoS_HOME}/data/mnist/csv
```

Then, train as follows:
```bash
# create a folder for new streaming data to arrive
export STREAM_DATA=${TFoS_HOME}/data/stream
mkdir -p ${STREAM_DATA}

# create a temp folder to stage streaming data
export TEMP_DATA=${TFoS_HOME}/data/tmp
mkdir -p ${TEMP_DATA}

# remove any old artifacts
rm -rf ${TFoS_HOME}/mnist_model
rm ${STREAM_DATA}/*
rm ${TEMP_DATA}/*

# start spark-streaming training job in background
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/examples/mnist/estimator/mnist_spark_streaming.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images_labels ${TFoS_HOME}/data/stream \
--model_dir ${TFoS_HOME}/mnist_model &

# save PID of spark job so we can stop it later
PID=$!

# wait for spark job to be RUNNING, then simulate arrival of NEW data in stream by:
# 1. making a copy of the data (to get a recent timestamp).
# 2. moving it into the stream folder atomically (to avoid spark picking up a partial file).
# for more info, see: http://spark.apache.org/docs/latest/streaming-programming-guide.html#basic-sources
# monitor the spark streaming logs after each command to view behavior.

COUNT=0

# wait for cluster to start
sleep 60
for f in ${TFoS_HOME}/data/mnist/csv/train/part-*; do cp $f ${TEMP_DATA}/$(basename $f | sed -e "s/[0-9][0-9]*/$COUNT/"); COUNT=$((COUNT + 1)); done; mv ${TEMP_DATA}/* ${STREAM_DATA}

# wait for next interval to add more files to the stream
sleep 60
for f in ${TFoS_HOME}/data/mnist/csv/train/part-*; do cp $f ${TEMP_DATA}/$(basename $f | sed -e "s/[0-9][0-9]*/$COUNT/"); COUNT=$((COUNT + 1)); done; mv ${TEMP_DATA}/* ${STREAM_DATA}

# wait for next interval to add more files to the stream
sleep 60
for f in ${TFoS_HOME}/data/mnist/csv/train/part-*; do cp $f ${TEMP_DATA}/$(basename $f | sed -e "s/[0-9][0-9]*/$COUNT/"); COUNT=$((COUNT + 1)); done; mv ${TEMP_DATA}/* ${STREAM_DATA}

# wait to finish processing all remaining files
sleep 120

# shutdown job via <ctrl-c>, `kill`, or `yarn application -kill <applicationId>`
kill $PID

# for a "graceful" shutdown, we provide the following tool to signal the SparkStreamingContext to stop.
# python ${TFoS_HOME}/examples/utils/stop_streaming.py <host> <port>
# Note: the host and port of the reservation server will be in the driver logs, e.g.
# "listening for reservations at ('127.0.0.1', 38254)"
```

#### Shutdown the Spark Standalone cluster
```bash
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

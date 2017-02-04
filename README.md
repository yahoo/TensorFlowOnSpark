<!--
Copyright 2017 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorFlowOnSpark

## Overview

TensorFlowOnSpark enables [TensorFlow](https://www.tensorflow.org) distributed training and inference on [Hadoop](http://hadoop.apache.org) clusters using [Apache Spark](http://spark.apache.org).  This framework seeks to minimize the amount of code changes required to run existing TensorFlow code on a shared grid.  It provides an Spark-compatible API to help manage the TensorFlow cluster with the following steps:

1. **Reservation** - reserves a port for the TensorFlow process on each executor and also starts a listener for data/control messages.
2. **Startup** - launches the Tensorflow main function on the executors.
3. **Data ingestion**
  1. **Feeding** - sends Spark RDD data into the TensorFlow nodes using the [feed_dict](https://www.tensorflow.org/how_tos/reading_data/#feeding) mechanism.  Note that we leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) for access to TFRecords on HDFS.
  2. **Readers & QueueRunners** - leverages TensorFlow's [Reader](https://www.tensorflow.org/how_tos/reading_data/#reading_from_files) mechanism to read data files directly from HDFS.
4. **Shutdown** - shuts down the Tensorflow workers and PS nodes on the executors.

We have also enhanced TensorFlow to support direct access to remote GPU memory (RDMA) on Infiniband networks.
This [enhancement](https://github.com/yahoo/tensorflow/tree/yahoo) address a [key issue](https://github.com/tensorflow/tensorflow/issues/2916)  of current TensorFlow network layer.

## Getting Started

Before you start, you should already be familiar with TensorFlow and have access to a Hadoop grid with Spark installed.  If your grid has GPU nodes, they must have cuda installed locally.

### Install Python 2.7

From your grid gateway, download/install Python into a local folder.  This installation of Python will be distributed to the Spark executors, so that any custom dependencies, including TensorFlow, will be available to the executors.

    # download and extract Python 2.7
    export PYTHON_ROOT=~/Python
    curl -O https://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
    tar -xvf Python-2.7.12.tgz
    rm Python-2.7.12.tgz

    # compile into local PYTHON_ROOT
    pushd Python-2.7.12
    ./configure --prefix="${PYTHON_ROOT}" --enable-unicode=ucs4
    make
    make install
    popd
    rm -rf Python-2.7.12

    # install pip
    pushd "${PYTHON_ROOT}"
    curl -O https://bootstrap.pypa.io/get-pip.py
    bin/python get-pip.py
    rm get-pip.py

    # install tensorflow (and any custom dependencies)
    ${PYTHON_ROOT}/bin/pip install pydoop
    # Note: add any extra dependencies here
    popd

### Install and compile TensorFlow w/ RDMA Support

    git clone git@github.com:yahoo/tensorflow.git
    # follow build instructions to install into ${PYTHON_ROOT}

### Install and compile Hadoop InputFormat/OutputFormat for TFRecords

    git clone https://github.com/tensorflow/ecosystem.git
    # follow build instructions to generate tensorflow-hadoop-1.0-SNAPSHOT.jar
    # copy jar to HDFS for easier reference
    hadoop fs -put tensorflow-hadoop-1.0-SNAPSHOT.jar

### Create a Python w/ TensorFlow zip package for Spark

    pushd "${PYTHON_ROOT}"
    zip -r Python.zip *
    popd

    # copy this Python distribution into HDFS
    hadoop fs -put ${PYTHON_ROOT}/Python.zip

### Install TensorFlowOnSpark

Next, clone this repo and build a zip package for Spark:

    git clone git@github.com:yahoo/TensorFlowOnSpark.git
    pushd TensorFlowOnSpark/src
    zip -r ../tfspark.zip *
    popd

## Run MNIST example

### Download/zip the MNIST dataset

    mkdir ${HOME}/mnist
    pushd ${HOME}/mnist >/dev/null
    curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    popd >/dev/null

### Convert the MNIST zip files into HDFS files

    # set environment variables (if not already done)
    export PYTHON_ROOT=~/Python
    export YROOT=~/y
    export LD_LIBRARY_PATH=${YROOT}/lib64:${PATH}
    export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
    export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
    export PATH=${PYTHON_ROOT}/bin/:$PATH
    export QUEUE=gpu
    
    # for CPU mode:
    # export QUEUE=default
    # remove --conf spark.executorEnv.LD_LIBRARY_PATH="lib64" \
    # remove --driver-library-path="lib64" \
    
    # save images and labels as CSV files
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 4G \
    --archives hdfs:///user/${USER}/Python.zip#Python,mnist/mnist.zip#mnist \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
    --output mnist/csv \
    --format csv

    # save images and labels as TFRecords
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 4G \
    --archives hdfs:///user/${USER}/Python.zip#Python,mnist/mnist.zip#mnist \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
    --output mnist/tfr \
    --format tfr

### Run distributed MNIST training (using feed_dict)

    # for CPU mode:
    # export QUEUE=default
    # remove --conf spark.executorEnv.LD_LIBRARY_PATH="lib64" \
    # remove --driver-library-path="lib64" \
    
    # hadoop fs -rm -r mnist_model
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 27G \
    --py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
    --images mnist/csv/train/images \
    --labels mnist/csv/train/labels \
    --mode train \
    --model mnist_model
    # to use infiniband, add --rdma

### Run distributed MNIST inference (using feed_dict)

    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 27G \
    --py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
    --images mnist/csv/test/images \
    --labels mnist/csv/test/labels \
    --mode inference \
    --model mnist_model \
    --output predictions

### Run distributed MNIST training (using QueueRunners)

    # for CPU mode:
    # export QUEUE=default
    # remove --conf spark.executorEnv.LD_LIBRARY_PATH="lib64" \
    # remove --driver-library-path="lib64" \

    # hadoop fs -rm -r mnist_model
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 27G \
    --py-files tensorflow/tfspark.zip,tensorflow/examples/mnist/tf/mnist_dist.py \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    tensorflow/examples/mnist/tf/mnist_spark.py \
    --images mnist/tfr/train \
    --format tfr \
    --mode train \
    --model mnist_model
    # to use infiniband, replace the last line with --model mnist_model --rdma

### Run distributed MNIST inference (using QueueRunners)

    # hadoop fs -rm -r predictions
    ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue ${QUEUE} \
    --num-executors 4 \
    --executor-memory 27G \
    --py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/tf/mnist_dist.py \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --archives hdfs:///user/${USER}/Python.zip#Python \
    --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64" \
    --driver-library-path="/usr/local/cuda-7.5/lib64" \
    TensorFlowOnSpark/examples/mnist/tf/mnist_spark.py \
    --images mnist/tfr/test \
    --mode inference \
    --model mnist_model \
    --output predictions

## Conversion Guide

The process of converting an existing TensorFlow application is fairly simple.  We highlight the main points here:

### Identify the main application

Every TensorFlow application will have a file containing a `main()` function and a call to `tf.app.run()`.  Locate this file first.

### Add PySpark and TensorFlowOnSpark imports

```python
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime
```

### Replace the main() function

Replace this line with `main_fun(argv, ctx)`.  The `argv` parameter will contain a full copy of the arguments supplied at the PySpark command line, while the `ctx` parameter will contain node metadata, like `job_name` and `task_id`.  Also, make sure that the `import tensorflow as tf` occurs within this function, since this will be executed/imported on the executors.  And, if there are any functions used by the main function, ensure that they are defined or imported inside the `main_fun` block.

```python
# def main():
def main_fun(argv, ctx)
  import tensorflow as tf
```

### Replace the tf.app.run() function

This line executes the TensorFlow main function.  Replace it with the following code to set up PySpark and launch TensorFlow on the executors.  Note that we're using `argparse` here mostly because the `tf.app.FLAGS` mechanism is currently not an officially supported TensorFlow API.

```python
if __name__ == '__main__':
    # tf.app.run()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
    args, rem = parser.parse_known_args()

    sc = SparkContext(conf=SparkConf().setAppName("your_app_name"))
    num_executors = int(sc._conf.get("spark.executor.instances"))
    num_ps = 1
    tensorboard = True

    cluster = TFCluster.reserve(sc, num_executors, num_ps, tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster.start(main_fun, sys.argv)
    cluster.shutdown()
```

### Replace the tf.train.Server() call

In distributed TensorFlow apps, there is typically code that:
1. extracts the addresses for the `ps` and `worker` nodes from the command line args
2. creates a cluster spec
3. starts the TensorFlow server.
These can all be replaced as follows.

```python
    # ps_hosts = FLAGS.ps_hosts.split(',')
    # worker_hosts = FLAGS.worker_hosts.split(',')
    # tf.logging.info('PS hosts are: %s' % ps_hosts)
    # tf.logging.info('Worker hosts are: %s' % worker_hosts)
    # cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    # server = tf.train.Server( {'ps': ps_hosts, 'worker': worker_hosts},
    #    job_name=FLAGS.job_name, task_index=FLAGS.task_id)
    cluster_spec, server = TFNode.start_cluster_server(ctx, FLAGS.num_gpus, FLAGS.rdma)
    # or use the following for default values of num_gpus=1 and rdma=False
    # cluster_spec, server = TFNode.start_cluster_server(ctx)
```

### Add TensorFlowOnSpark-specific arguments

Since most TensorFlow examples use the `tf.app.FLAGS` mechanism, we leverage it here to parse our TensorFlowOnSpark-specific arguments (on the executor-side) for consistency.  If your application uses another parsing mechanism, just add these two arguments accordingly.

```python
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs per node.')
tf.app.flags.DEFINE_boolean('rdma', False, 'Use RDMA between GPUs')
```

Note: while these are required for the `TFNode.start_cluster_server()` function, your code must still be written specifically to leverage multiple GPUs (e.g. see the "tower" pattern in the CIFAR-10 example).  And again, if using a single GPU per node with no RDMA, you can skip this step and just use `TFNode.start_cluster_server(ctx).

### Enable TensorBoard

Finally, if using TensorBoard, ensure that the summaries are saved to the local disk of the "chief" worker (by convention "worker:0"), since TensorBoard currently cannot read directly from HDFS.  Locate the `tf.train.Supervisor()` call, and add a custom `summary_writer` as follows.  Note: the tensorboard process will looking in this specific directory by convention, so do not change the path.

```python
  summary_writer = tf.summary.FileWriter("tensorboard_%d" %(ctx.worker_num), graph=tf.get_default_graph())
  sv = tf.train.Supervisor(is_chief=is_chief,
                           logdir=FLAGS.train_dir,
                           init_op=init_op,
                           summary_op=None,
                           global_step=global_step,
                           summary_writer=summary_writer,
                           saver=saver,
                           save_model_secs=FLAGS.save_interval_secs)
```

### Try it out

Using a similar PySpark command as the MNIST example above, you should be now able to launch your job on the grid.

## Other Examples

In addition to the MNIST example, we also demonstrate the conversion process on several of the TensorFlow examples:

- [CIFAR10](examples/cifar10)
- [ImageNet/Inception](examples/imagenet/inception)
- [ImageNet/Inception (using TF-Slim)](examples/slim)

## For more information

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow MOOC on Udacity] (https://www.udacity.com/course/deep-learning--ud730)

## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.

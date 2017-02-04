<!--
Copyright 2017 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorFlowOnSpark

## Overview

TensorFlowOnSpark enables [TensorFlow](https://www.tensorflow.org) distributed training and inference on [Hadoop](http://hadoop.apache.org) clusters using [Apache Spark](http://spark.apache.org).  This framework seeks to minimize the amount of code changes required to run existing TensorFlow code on a shared grid.  

TensorFlowOnSpark provides an Spark-compatible API to help manage the TensorFlow cluster with the following steps:

1. **Reservation** - reserves a port for the TensorFlow process on each executor and also starts a listener for data/control messages.
2. **Startup** - launches the Tensorflow main function on the executors.
3. **Data ingestion**
  1. **Feeding** - sends Spark RDD data into the TensorFlow nodes using the [feed_dict](https://www.tensorflow.org/how_tos/reading_data/#feeding) mechanism.  Note that we leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) for access to TFRecords on HDFS.
  2. **Readers & QueueRunners** - leverages TensorFlow's [Reader](https://www.tensorflow.org/how_tos/reading_data/#reading_from_files) mechanism to read data files directly from HDFS.
4. **Shutdown** - shuts down the Tensorflow workers and PS nodes on the executors.

We have also enhanced TensorFlow to support direct access to remote GPU memory (RDMA) on Infiniband networks.
This [enhancement](https://github.com/yahoo/tensorflow/tree/yahoo) address a [key issue](https://github.com/tensorflow/tensorflow/issues/2916)  of current TensorFlow network layer.


## Using TensorFlowOnSpark

Please check TensorFlowOnSpark [wiki site](../../wiki) for detailed
documentations such as getting started guides for [standalone
cluster](../../wiki/GetStarted_local) and [AWS EC2
cluster](../../wiki/GetStarted_EC2).

Please join [TensorFlowOnSpark user group](https://groups.google.com/forum/#!forum/TensorFlowOnSpark-users) for discussions and questions.

## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.

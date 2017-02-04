<!--
Copyright 2017 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorFlowOnSpark

## Overview

TensorFlowOnSpark enables [TensorFlow](https://www.tensorflow.org) distributed training and inference on [Hadoop](http://hadoop.apache.org) clusters using [Apache Spark](http://spark.apache.org).  This framework seeks to minimize the amount of code changes required to run existing TensorFlow code on a shared grid.  It provides an Spark-compatible API to help manage the TensorFlow cluster with the following steps:

1. **Reservation** - reserves a port for the TensorFlow process on each executor and also starts to listen for data/control messages.
2. **Startup** - launches the Tensorflow main function on the executors.
3. **Data ingestion**
  a. **Feeding** - sends Spark RDD data into the TensorFlow nodes using the [feed_dict](https://www.tensorflow.org/how_tos/reading_data/#feeding) mechanism. We leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) for access to TFRecords on HDFS.
  b. **Readers & QueueRunners** - leverages TensorFlow's [Reader](https://www.tensorflow.org/how_tos/reading_data/#reading_from_files) mechanism to read data files.
4. **Shutdown** - shutdown Tensorflow execution on executors.

We have enhanced TensorFlow to support direct access to remote GPU memory (RDMA) on Infiniband networks.  
This [enhancement](https://github.com/yahoo/tensorflow/tree/yahoo) address a [key issue](https://github.com/tensorflow/tensorflow/issues/2916)  of current TensorFlow network layer.


## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.

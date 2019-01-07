<!--
Copyright 2017 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorFlowOnSpark
> _TensorFlowOnSpark brings scalable deep learning to Apache Hadoop and Apache Spark
clusters._

[![Build Status](https://travis-ci.org/yahoo/TensorFlowOnSpark.svg?branch=master)](https://travis-ci.org/yahoo/TensorFlowOnSpark) [![PyPI version](https://badge.fury.io/py/tensorflowonspark.svg)](https://badge.fury.io/py/tensorflowonspark)

By combining salient features from the [TensorFlow](https://www.tensorflow.org) deep learning framework with [Apache Spark](http://spark.apache.org) and [Apache Hadoop](http://hadoop.apache.org), TensorFlowOnSpark enables distributed
deep learning on a cluster of GPU and CPU servers.

It enables both distributed TensorFlow training and
inferencing on Spark clusters, with a goal to minimize the amount
of code changes required to run existing TensorFlow programs on a
shared grid.  Its Spark-compatible API helps manage the TensorFlow
cluster with the following steps:

1. **Startup** - launches the Tensorflow main function on the executors, along with listeners for data/control messages.
1. **Data ingestion**
   - **InputMode.TENSORFLOW** - leverages TensorFlow's built-in APIs to read data files directly from HDFS.
   - **InputMode.SPARK** - sends Spark RDD data to the TensorFlow nodes via the [feed_dict](https://www.tensorflow.org/how_tos/reading_data/#feeding) mechanism.  Note that we leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) to access TFRecords on HDFS.
1. **Shutdown** - shuts down the Tensorflow workers and PS nodes on the executors.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contribute](#contribute)
- [License](#license)

## Background

TensorFlowOnSpark was developed by Yahoo for large-scale distributed
deep learning on our Hadoop clusters in Yahoo's private cloud. 

TensorFlowOnSpark provides some important benefits (see [our
blog](http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep))
over alternative deep learning solutions.
   * Easily migrate all existing TensorFlow programs with <10 lines of code change;
   * Support all TensorFlow functionalities: synchronous/asynchronous training, model/data parallelism, inferencing and TensorBoard;
   * Server-to-server direct communication achieves faster learning when available;
   * Allow datasets on HDFS and other sources pushed by Spark or pulled by TensorFlow; 
   * Easily integrate with your existing data processing pipelines and machine learning algorithms (ex. MLlib, CaffeOnSpark);
   * Easily deployed on cloud or on-premise: CPU & GPU, Ethernet and Infiniband. 

## Install

TensorFlowOnSpark is provided as a pip package, which can be installed on single machines via:
```
pip install tensorflowonspark
```

For distributed clusters, please see our [wiki site](../../wiki) for detailed documentation for specific environments, such as our getting started guides for [single-node Spark Standalone](https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_Standalone), [YARN clusters](../../wiki/GetStarted_YARN) and [AWS EC2](../../wiki/GetStarted_EC2).  Note: the Windows operating system is not currently supported due to [this issue](https://github.com/yahoo/TensorFlowOnSpark/issues/36).

## Usage

To use TensorFlowOnSpark with an existing TensorFlow application, you can follow our [Conversion Guide](../../wiki/Conversion-Guide) to describe the required changes.  Additionally, our [wiki site](../../wiki) has pointers to some presentations which provide an overview of the platform.

## API

[API Documentation](https://yahoo.github.io/TensorFlowOnSpark/) is automatically generated from the code.

## Contribute

Please join the [TensorFlowOnSpark user group](https://groups.google.com/forum/#!forum/TensorFlowOnSpark-users) for discussions and questions.  If you have a question, please review our [FAQ](../../wiki/Frequently-Asked-Questions) before posting.

Contributions are always welcome.  For more information, please see our [guide for getting involved](Contributing.md).


## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.

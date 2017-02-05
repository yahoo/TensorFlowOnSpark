<!--
Copyright 2017 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# TensorFlowOnSpark

## What's TensorFlowOnSpark?

TensorFlowOnSpark brings scalable deep learning to Hadoop and Spark
clusters. By combining salient features from deep learning framework
[TensorFlow](https://www.tensorflow.org) and big-data frameworks
Apache Spark and Apache Hadoop, TensorFlowOnSpark enables distributed
deep learning on a cluster of GPU and CPU servers.

TensorFlowOnSpark enables distributed
[TensorFlow](https://www.tensorflow.org) training and inference on
[Apache Spark](http://spark.apache.org) clusters.  It seeks to
minimize the amount of code changes required to run existing
TensorFlow programs on a shared grid.  Its Spark-compatible API helps
manage the TensorFlow cluster with the following steps:

1. **Reservation** - reserves a port for the TensorFlow process on each executor and also starts a listener for data/control messages.
2. **Startup** - launches the Tensorflow main function on the executors.
3. **Data ingestion**
  1. **Feeding** - sends Spark RDD data into the TensorFlow nodes using the [feed_dict](https://www.tensorflow.org/how_tos/reading_data/#feeding) mechanism.  Note that we leverage the [Hadoop Input/Output Format](https://github.com/tensorflow/ecosystem/tree/master/hadoop) for access to TFRecords on HDFS.
  2. **Readers & QueueRunners** - leverages TensorFlow's [Reader](https://www.tensorflow.org/how_tos/reading_data/#reading_from_files) mechanism to read data files directly from HDFS.
4. **Shutdown** - shuts down the Tensorflow workers and PS nodes on the executors.

We have also
[enhanced](https://github.com/yahoo/tensorflow/tree/yahoo) TensorFlow
to support direct access to remote memory (RDMA) on Infiniband
networks.

TensorFlowOnSpark was developed by Yahoo for large-scale distributed
deep learning on our Hadoop clusters in Yahoo's private cloud. 


## Why TensorFlowOnSpark?

TensorFlowOnSpark provides some important benefits (see [our
blog](https://docs.google.com/a/yahoo-inc.com/document/d/16IqUa7A3mRc868D6jH82Wos4cqYg7wg-P_1cC7o23Qs/edit?usp=sharing))
over alternative deep learning solutions.
   * Easily migrate all existing TensorFlow programs with <10 lines of code change;
   * Support all TensorFlow functionalities: synchronous/asynchronous training, model/data parallelism, inferencing and TensorBoard;
   * Server-to-server direct communication achieves faster learning when available;
   * Allow datasets on HDFS and other sources pushed by Spark or pulled by TensorFlow; 
   * Easily integrate with your existing data processing pipelines and machine learning algorithms (ex. MLlib, CaffeOnSpark);
   * Easily deployed on cloud or on-premise: CPU & GPU, Ethernet and Infiniband. 


## Using TensorFlowOnSpark

Please check TensorFlowOnSpark [wiki site](../../wiki) for detailed
documentations such as getting started guides for [YARN
cluster](../../wiki/GetStarted_YARN) and [AWS EC2
cluster](../../wiki/GetStarted_EC2). A [Conversion
Guide](../../wiki/Conversion) has been provided to help you convert
your TensorFlow programs.

Please join [TensorFlowOnSpark user group](https://groups.google.com/forum/#!forum/TensorFlowOnSpark-users) for discussions and questions.

## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See [LICENSE](LICENSE) file for terms.

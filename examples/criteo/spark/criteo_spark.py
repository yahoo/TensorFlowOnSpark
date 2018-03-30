# Copyright 2018 Criteo
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed Criteo Display CTR prediction on grid based on TensorFlow on Spark
# https://github.com/yahoo/TensorFlowOnSpark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
from datetime import datetime



from tensorflowonspark import TFCluster


import criteo_dist


if __name__ == "__main__":
    sc = SparkContext(conf=SparkConf().setAppName("criteo_spark"))
    executors = sc._conf.get("spark.executor.instances")
    if executors is None:
        raise Exception("Could not retrieve the number of executors from the SparkContext")
    num_executors = int(executors)
    num_ps = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
    parser.add_argument("-i", "--data", help="HDFS path to data in parallelized format")
    parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="criteo_model")
    parser.add_argument("-v", "--validation", help="HDFS path to validation data")

    parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
    parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=1)
    parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000)
    parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("-X", "--mode", help="train|inference", default="train")
    parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
    parser.add_argument("-tbld", "--tensorboardlogdir",
                        help="Tensorboard log directory. It should on hdfs. Thus, it must be prefixed with hdfs://default")

    args = parser.parse_args()
    print("args:", args)

    print("{0} ===== Start".format(datetime.now().isoformat()))

    dataRDD = sc.textFile(args.data).map(lambda ln: [x for x in ln.split('\t')])

    cluster = TFCluster.run(sc, criteo_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                            TFCluster.InputMode.SPARK, log_dir=args.model)
    if args.mode == "train":
        cluster.train(dataRDD, args.epochs)
    else:
        labelRDD = cluster.inference(dataRDD)
        labelRDD.saveAsTextFile(args.output)
    cluster.shutdown()
    print("{0} ===== Stop".format(datetime.now().isoformat()))
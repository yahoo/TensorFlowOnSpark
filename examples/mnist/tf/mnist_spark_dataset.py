# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
from datetime import datetime

from tensorflowonspark import TFCluster
import mnist_dist_dataset

sc = SparkContext(conf=SparkConf().setAppName("mnist_tf"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size",
                    help="number of records per batch", type=int, default=100)
parser.add_argument(
    "-e", "--epochs", help="number of epochs", type=int, default=0)
parser.add_argument("-f", "--format", help="example format: (csv2|tfr)",
                    choices=["csv2", "tfr"], default="tfr")
parser.add_argument(
    "-i", "--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument(
    "-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument(
    "-m", "--model", help="HDFS path to save/load model during train/test", default="mnist_model")
parser.add_argument("-n", "--cluster_size",
                    help="number of nodes in the cluster (for Spark Standalone)", type=int, default=num_executors)
parser.add_argument(
    "-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument(
    "-r", "--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument(
    "-s", "--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("-tb", "--tensorboard",
                    help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
parser.add_argument("-p", "--driver_ps_nodes", help="""run tensorflow PS node on driver locally.
    You will need to set cluster_size = num_executors + num_ps""", default=False)
args = parser.parse_args()
print("args:", args)


print("{0} ===== Start".format(datetime.now().isoformat()))
cluster = TFCluster.run(sc, mnist_dist_dataset.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                        TFCluster.InputMode.TENSORFLOW, driver_ps_nodes=args.driver_ps_nodes)
cluster.shutdown()

print("{0} ===== Stop".format(datetime.now().isoformat()))

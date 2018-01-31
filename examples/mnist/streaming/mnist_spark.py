# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.streaming import StreamingContext

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflowonspark import TFCluster
import mnist_dist

sc = SparkContext(conf=SparkConf().setAppName("mnist_streaming"))
ssc = StreamingContext(sc, 10)
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size",
                    help="number of records per batch", type=int, default=100)
parser.add_argument(
    "-e", "--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("-f", "--format", help="example format: (csv|csv2|pickle|tfr)",
                    choices=["csv", "csv2", "pickle", "tfr"], default="stream")
parser.add_argument(
    "-i", "--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument(
    "-m", "--model", help="HDFS path to save/load model during train/inference", default="mnist_model")
parser.add_argument("-n", "--cluster_size",
                    help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument(
    "-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument(
    "-s", "--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("-tb", "--tensorboard",
                    help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()
print("args:", args)

print("{0} ===== Start".format(datetime.now().isoformat()))


def parse(ln):
  lbl, img = ln.split('|')
  image = [int(x) for x in img.split(',')]
  label = numpy.zeros(10)
  label[int(lbl)] = 1.0
  return (image, label)


stream = ssc.textFileStream(args.images)
imageRDD = stream.map(lambda ln: parse(ln))

cluster = TFCluster.run(sc, mnist_dist.map_fun, args, args.cluster_size,
                        num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
if args.mode == "train":
  cluster.train(imageRDD)
else:
  labelRDD = cluster.inference(imageRDD)
  labelRDD.saveAsTextFiles(args.output)

ssc.start()
cluster.shutdown(ssc)

print("{0} ===== Stop".format(datetime.now().isoformat()))

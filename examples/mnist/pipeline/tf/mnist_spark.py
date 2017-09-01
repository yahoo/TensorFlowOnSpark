# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

import argparse
from datetime import datetime

from tensorflowonspark import TFCluster, dfutil
from tensorflowonspark.pipeline import TFEstimator
import mnist_dist

sc = SparkContext(conf=SparkConf().setAppName("mnist_tf"))
spark = SparkSession(sc)

executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()

######## PARAMS ########

## TFoS/cluster
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("--model_dir", help="HDFS path to save/load model during train/inference", type=str)
parser.add_argument("--export_dir", help="HDFS path to export model", type=str)
parser.add_argument("--tfr_dir", help="HDFS path to temporarily save DataFrame to disk", type=str)
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--num_ps", help="number of PS nodes in cluster", type=int, default=1)
parser.add_argument("--protocol", help="Tensorflow network protocol (grpc|rdma)", default="grpc")
parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

######## ARGS ########

# Spark input/output
parser.add_argument("--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
parser.add_argument("--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("--output", help="HDFS path to save test/inference output", default="predictions")

args = parser.parse_args()
print("args:",args)

images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')])
labels = sc.textFile(args.labels).map(lambda ln: [int(float(x)) for x in ln.split(',')])
dataRDD = images.zip(labels)
df = spark.createDataFrame(dataRDD, ['image', 'label'])

print("{0} ===== Start".format(datetime.now().isoformat()))
estimator = TFEstimator(mnist_dist.map_fun, args, export_fn=mnist_dist.export_fun) \
        .setModelDir(args.model_dir) \
        .setExportDir(args.export_dir) \
        .setClusterSize(args.cluster_size) \
        .setNumPS(args.num_ps) \
        .setInputMode(TFCluster.InputMode.TENSORFLOW) \
        .setProtocol(args.protocol) \
        .setReaders(args.readers) \
        .setTensorboard(args.tensorboard) \
        .setEpochs(args.epochs) \
        .setBatchSize(args.batch_size) \
        .setSteps(args.steps)

tf_args = { 'initial_learning_rate': 0.01 }
model = estimator.fit(df)

print("{0} ===== Stop".format(datetime.now().isoformat()))


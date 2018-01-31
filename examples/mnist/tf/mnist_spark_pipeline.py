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
import sys
import tensorflow as tf
from datetime import datetime

from tensorflowonspark import TFCluster, dfutil
from tensorflowonspark.pipeline import TFEstimator, TFModel
import mnist_dist_pipeline

sc = SparkContext(conf=SparkConf().setAppName("mnist_tf"))
spark = SparkSession(sc)

executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()

######## ARGS ########

# TFoS/cluster
parser.add_argument(
    "--batch_size", help="number of records per batch", type=int, default=100)
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument(
    "--model_dir", help="HDFS path to save/load model during train/inference", type=str)
parser.add_argument("--export_dir", help="HDFS path to export model", type=str)
parser.add_argument(
    "--tfrecord_dir", help="HDFS path to temporarily save DataFrame to disk", type=str)
parser.add_argument("--cluster_size", help="number of nodes in the cluster",
                    type=int, default=num_executors)
parser.add_argument(
    "--num_ps", help="number of PS nodes in cluster", type=int, default=1)
parser.add_argument("-p", "--driver_ps_nodes", help="""run tensorflow PS node on driver locally.
    You will need to set cluster_size = num_executors + num_ps""", default=False)
parser.add_argument(
    "--protocol", help="Tensorflow network protocol (grpc|rdma)", default="grpc")
parser.add_argument(
    "--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument(
    "--steps", help="maximum number of steps", type=int, default=1000)
parser.add_argument(
    "--tensorboard", help="launch tensorboard process", action="store_true")

# Spark input/output
parser.add_argument("--format", help="example format: (csv|tfr)",
                    choices=["csv", "tfr"], default="csv")
parser.add_argument(
    "--images", help="HDFS path to MNIST images in parallelized format")
parser.add_argument(
    "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument(
    "--output", help="HDFS path to save test/inference output", default="predictions")

# Execution Modes
parser.add_argument(
    "--train", help="train a model using Estimator", action="store_true")
parser.add_argument("--inference_mode", help="type of inferencing (none|signature|direct)",
                    choices=["none", "signature", "direct"], default="none")
parser.add_argument("--inference_output", help="output of inferencing (predictions|features)",
                    choices=["predictions", "features"], default="predictions")

args = parser.parse_args()
print("args:", args)

print("{0} ===== Start".format(datetime.now().isoformat()))

if args.format == "tfr":
  df = dfutil.loadTFRecords(sc, args.images)
elif args.format == "csv":
  images = sc.textFile(args.images).map(
      lambda ln: [int(x) for x in ln.split(',')])
  labels = sc.textFile(args.labels).map(
      lambda ln: [int(float(x)) for x in ln.split(',')])
  dataRDD = images.zip(labels)
  df = spark.createDataFrame(dataRDD, ['image', 'label'])
else:
  raise Exception("Unsupported format: {}".format(args.format))

# Pipeline API

if args.train:
  # train a model using Spark Estimator fitted to a DataFrame
  print("{0} ===== Estimator.fit()".format(datetime.now().isoformat()))
  # dummy tf args (from imagenet/inception example)
  tf_args = {'initial_learning_rate': 0.045,
             'num_epochs_per_decay': 2.0, 'learning_rate_decay_factor': 0.94}
  estimator = TFEstimator(mnist_dist_pipeline.map_fun, args, export_fn=mnist_dist_pipeline.export_fun) \
      .setModelDir(args.model_dir) \
      .setExportDir(args.export_dir) \
      .setClusterSize(args.cluster_size) \
      .setNumPS(args.num_ps) \
      .setDriverPSNodes(args.driver_ps_nodes) \
      .setInputMode(TFCluster.InputMode.TENSORFLOW) \
      .setTFRecordDir(args.tfrecord_dir) \
      .setProtocol(args.protocol) \
      .setReaders(args.readers) \
      .setTensorboard(args.tensorboard) \
      .setEpochs(args.epochs) \
      .setBatchSize(args.batch_size) \
      .setSteps(args.steps)
  model = estimator.fit(df)
else:
  # use a previously trained/exported model
  model = TFModel(args) \
      .setExportDir(args.export_dir) \
      .setBatchSize(args.batch_size)

# NO INFERENCING
if args.inference_mode == 'none':
  sys.exit(0)

# INFER USING EXPORTED SIGNATURES OF TENSORFLOW SAVED_MODEL
elif args.inference_mode == 'signature':
  model.setModelDir(None)
  # load saved_model from args.export_dir
  model.setExportDir(args.export_dir)
  # using default SERVING tagset
  model.setTagSet(tf.saved_model.tag_constants.SERVING)
  # map DataFrame 'image' column to the 'image' input tensor alias of signature
  model.setInputMapping({'image': 'image'})
  if args.inference_output == 'predictions':
    # default signature def key, i.e. 'predict'
    model.setSignatureDefKey(
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    # map 'prediction' output tensor alias to output DataFrame 'col_out' column
    model.setOutputMapping({'prediction': 'col_out'})
  else:  # args.inference_output == 'features'
    # custom signature def key
    model.setSignatureDefKey('featurize')
    # map 'features' output tensor alias to output DataFrame 'col_out' column
    model.setOutputMapping({'features': 'col_out'})

# INFER USING TENSORFLOW SAVED_MODEL, IGNORING EXPORTED SIGNATURES
else:  # args.inference_mode == 'direct':
  model.setModelDir(None)
  # load saved_model from args.export_dir
  model.setExportDir(args.export_dir)
  # using default SERVING tagset
  model.setTagSet(tf.saved_model.tag_constants.SERVING)
  # map DataFrame 'image' column to the 'x' input tensor
  model.setInputMapping({'image': 'x'})
  if args.inference_output == 'predictions':
    # map 'prediction' output tensor to output DataFrame 'col_out' column
    model.setOutputMapping({'prediction': 'col_out'})
  else:  # args.inference_output == 'features'
    # add 'Relu' output tensor to output DataFrame 'col_out2' column
    model.setOutputMapping({'prediction': 'col_out', 'Relu': 'col_out2'})

print("{0} ===== Model.transform()".format(datetime.now().isoformat()))
preds = model.transform(df)
preds.write.json(args.output)

print("{0} ===== Stop".format(datetime.now().isoformat()))

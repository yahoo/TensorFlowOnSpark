# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster, TFNode, dfutil
from tensorflowonspark.pipeline import TFEstimator
from datetime import datetime

from inception import inception_export

import sys
import tensorflow as tf

def main_fun(argv, ctx):
  # extract node metadata from ctx
  job_name = ctx.job_name
  task_index = ctx.task_index

  assert job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  from inception import inception_distributed_train
  from inception.imagenet_data import ImagenetData
  import tensorflow as tf

  # instantiate FLAGS on workers using argv from driver and add job_name and task_id
  print("argv:", argv)
  sys.argv = argv

  FLAGS = tf.app.flags.FLAGS
  FLAGS.job_name = job_name
  FLAGS.task_id = task_index
  print("FLAGS:", FLAGS.__dict__['__flags'])

  # Get TF cluster and server instances
  cluster_spec, server = TFNode.start_cluster_server(ctx, FLAGS.num_gpus, FLAGS.rdma)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec, ctx)


if __name__ == '__main__':
  # parse arguments needed by the Spark driver
  import argparse

  sc = SparkContext(conf=SparkConf().setAppName('imagenet_distributed_train'))
  spark = SparkSession.builder.getOrCreate()
  num_executors = int(sc._conf.get("spark.executor.instances"))

  # Note: these arguments are for TFoS only... since the Inception code uses tf.app.FLAGS, for which we need to pass the argv
  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--export_dir", help="HDFS path to export model", type=str)
  parser.add_argument("--input_mode", help="method to ingest data: (spark|tf)", choices=["spark","tf"], default="tf")
  parser.add_argument("--num_ps", help="number of PS nodes in cluster", type=int, default=1)
  parser.add_argument("--output", help="HDFS path to save output predictions", type=str)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--train_dir", help="HDFS path to save/load model during train/inference", type=str)
  parser.add_argument("--tfrecord_dir", help="HDFS path to temporarily save DataFrame to disk", type=str)
  parser.add_argument("--train_data", help="HDFS path to training data", type=str)
  parser.add_argument("--validation_data", help="HDFS path to validation data", type=str)

  (args,rem) = parser.parse_known_args()

  input_mode = TFCluster.InputMode.SPARK if args.input_mode == 'spark' else TFCluster.InputMode.TENSORFLOW

  print("{0} ===== Start".format(datetime.now().isoformat()))

  df = dfutil.loadTFRecords(sc, args.train_data, binary_features=['image/encoded'])
  estimator = TFEstimator(main_fun, sys.argv, export_fn=inception_export.export) \
          .setModelDir(args.train_dir) \
          .setExportDir(args.export_dir) \
          .setTFRecordDir(args.tfrecord_dir) \
          .setClusterSize(args.cluster_size) \
          .setNumPS(args.num_ps) \
          .setInputMode(TFCluster.InputMode.TENSORFLOW) \
          .setTensorboard(args.tensorboard) \

  print("{0} ===== Train".format(datetime.now().isoformat()))
  model = estimator.fit(df)

  print("{0} ===== Inference".format(datetime.now().isoformat()))
  df = dfutil.loadTFRecords(sc, args.validation_data, binary_features=['image/encoded'])
  preds = model.setTagSet(tf.saved_model.tag_constants.SERVING) \
              .setSignatureDefKey(tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY) \
              .setInputMapping({'image/encoded': 'jpegs', 'image/class/label': 'labels'}) \
              .setOutputMapping({'top_5_acc': 'output'}) \
              .transform(df)
  preds.write.json(args.output)

  print("{0} ===== Stop".format(datetime.now().isoformat()))

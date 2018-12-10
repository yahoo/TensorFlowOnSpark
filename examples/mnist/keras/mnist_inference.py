# Copyright 2018 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# This example demonstrates how to leverage Spark for parallel inferencing from a SavedModel.
#
# Normally, you can use TensorFlowOnSpark to just form a TensorFlow cluster for training and inferencing.
# However, in some situations, you may have a SavedModel without the original code for defining the inferencing
# graph.  In these situations, we can use Spark to instantiate a single-node TensorFlow instance on each executor,
# where each executor can independently load the model and inference on input data.
#
# Note: this particular example demonstrates use of `tf.data.Dataset` to read the input data for inferencing, 
# but it could also be adapted to just use an RDD of TFRecords from Spark.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

IMAGE_PIXELS = 28


def inference(it, num_workers, args):
  from tensorflowonspark import util

  # consume worker number from RDD partition iterator
  for i in it:
    worker_num = i
  print("worker_num: {}".format(i))

  # setup env for single-node TF
  util.single_node_env()

  # load saved_model using default tag and signature
  sess = tf.Session()
  tf.saved_model.loader.load(sess, ['serve'], args.export)

  # parse function for TFRecords
  def parse_tfr(example_proto):
    feature_def = {"label": tf.FixedLenFeature(10, tf.int64),
                   "image": tf.FixedLenFeature(IMAGE_PIXELS * IMAGE_PIXELS, tf.int64)}
    features = tf.parse_single_example(example_proto, feature_def)
    norm = tf.constant(255, dtype=tf.float32, shape=(784,))
    image = tf.div(tf.to_float(features['image']), norm)
    label = tf.to_float(features['label'])
    return (image, label)

  # define a new tf.data.Dataset (for inferencing)
  ds = tf.data.Dataset.list_files("{}/part-*".format(args.images_labels))
  ds = ds.shard(num_workers, worker_num)
  ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=1)
  ds = ds.map(parse_tfr).batch(10)
  iterator = ds.make_one_shot_iterator()
  image_label = iterator.get_next(name='inf_image')

  # create an output file per spark worker for the predictions
  tf.gfile.MakeDirs(args.output)
  output_file = tf.gfile.GFile("{}/part-{:05d}".format(args.output, worker_num), mode='w')

  while True:
    try:
      # get images and labels from tf.data.Dataset
      img, lbl = sess.run(['inf_image:0', 'inf_image:1'])

      # inference by feeding these images and labels into the input tensors
      # you can view the exported model signatures via:
      #     saved_model_cli show --dir <export_dir> --all

      # note that we feed directly into the graph tensors (bypassing the exported signatures)
      # these tensors will be shown in the "name" field of the signature definitions

      outputs = sess.run(['dense_2/Softmax:0'], feed_dict={'Placeholder:0': img})
      for p in outputs[0]:
        output_file.write("{}\n".format(np.argmax(p)))
    except tf.errors.OutOfRangeError:
      break

  output_file.close()


if __name__ == '__main__':
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  sc = SparkContext(conf=SparkConf().setAppName("mnist_inference"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster (for S with labelspark Standalone)", type=int, default=num_executors)
  parser.add_argument('--images_labels', type=str, help='Directory for input images with labels')
  parser.add_argument("--export", help="HDFS path to export model", type=str, default="mnist_export")
  parser.add_argument("--output", help="HDFS path to save predictions", type=str, default="predictions")
  args, _ = parser.parse_known_args()
  print("args: {}".format(args))

  # Not using TFCluster... just running single-node TF instances on each executor
  nodes = list(range(args.cluster_size))
  nodeRDD = sc.parallelize(list(range(args.cluster_size)), args.cluster_size)
  nodeRDD.foreachPartition(lambda worker_num: inference(worker_num, args.cluster_size, args))

# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.ml.param.shared import *
from pyspark.ml.pipeline import Estimator, Model, Pipeline
from pyspark.sql import SparkSession

import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader, signature_def_utils
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader
from . import TFCluster, gpu_info

import logging
import os
import subprocess

class TFEstimator(Estimator):
  """Spark ML Pipeline Estimator which launches a TensorFlowOnSpark cluster for training"""
  train_fn = None
  args = None
  input_mode = None

  def __init__(self, train_fn, args, input_mode):
    self.train_fn = train_fn
    self.args = args
    self.input_mode = input_mode

  def _fit(self, dataset):
    self.args.mode = 'train'
    logging.info("===== train args: {0}".format(self.args))
    sc = SparkContext.getOrCreate()
    cluster = TFCluster.run(sc, self.train_fn, self.args, self.args.cluster_size, self.args.num_ps, self.args.tensorboard, self.input_mode)
    cluster.train(dataset.rdd, self.args.epochs)
    cluster.shutdown()
    return TFModel(self.args)

class TFModel(Model):
  """Spark ML Pipeline Model which runs a TensorFlow SavedModel stored on disk."""
  args = None

  def __init__(self, args):
    self.args = args

  def _transform(self, dataset):
    logging.info("===== inference args: {0}".format(self.args))
    spark = SparkSession.builder.getOrCreate()
    rdd_out = dataset.rdd.mapPartitions(lambda it: _run_saved_model(it, self.args))
    return spark.createDataFrame(rdd_out, "string")

def _run_saved_model(iterator, args):
  """
  Run a SavedModel using a single input tensor obtained from a Spark partition iterator and return a single output tensor.
  Based on https://github.com/tensorflow/tensorflow/blob/8e0e8d41a3a8f2d4a6100c2ea1dc9d6c6c4ad382/tensorflow/python/tools/saved_model_cli.py#L233
  """
  # ensure expanded CLASSPATH w/o glob characters (required for Spark 2.1 + JNI)
  if 'HADOOP_PREFIX' in os.environ and 'TFOS_CLASSPATH_UPDATED' not in os.environ:
      classpath = os.environ['CLASSPATH']
      hadoop_path = os.path.join(os.environ['HADOOP_PREFIX'], 'bin', 'hadoop')
      hadoop_classpath = subprocess.check_output([hadoop_path, 'classpath', '--glob']).decode()
      logging.debug("CLASSPATH: {0}".format(hadoop_classpath))
      os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath
      os.environ['TFOS_CLASSPATH_UPDATED'] = '1'

  logging.info("===== loading meta_graph_def for tag_set ({0}) from {1}".format(args.tag_set, args.export_dir))
  meta_graph_def = get_meta_graph_def(args.export_dir, args.tag_set)
  inputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, args.signature_def_key).inputs
  logging.debug("inputs_tensor_info: {0}".format(inputs_tensor_info))
  outputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, args.signature_def_key).outputs
  logging.debug("outputs_tensor_info: {0}".format(outputs_tensor_info))

  logging.info("===== creating single-node session")
  if tf.test.is_built_with_cuda():
    # GPU
    num_gpus = args.num_gpus if 'num_gpus' in args else 1
    gpus_to_use = gpu_info.get_gpus(num_gpus)
    logging.info("Using gpu(s): {0}".format(gpus_to_use))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
  else:
    # CPU
    logging.info("Using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  # Note: if there is a GPU conflict (CUDA_ERROR_INVALID_DEVICE), the entire task will fail and retry.
  result = []
  logging.info("===== running saved_model")
  with tf.Session(graph=ops_lib.Graph()) as sess:
    loader.load(sess, args.tag_set.split(','), args.export_dir)
    for batch in yield_batch(iterator, args.batch_size):
      # batch type must match the input_tensor type
      inputs_feed_dict = {
        inputs_tensor_info[args.tensor_in].name: batch
      }
      output_tensor_names = [outputs_tensor_info[args.tensor_out].name]
      output_tensors = sess.run(output_tensor_names, feed_dict=inputs_feed_dict)
      outputs = [x.item() for x in output_tensors[0]]               # convert from numpy to standard python types
      result.extend(outputs)
  return result

def get_meta_graph_def(saved_model_dir, tag_set):
  """
  Utility function to read a meta_graph_def from disk.
  From https://github.com/tensorflow/tensorflow/blob/8e0e8d41a3a8f2d4a6100c2ea1dc9d6c6c4ad382/tensorflow/python/tools/saved_model_cli.py#L186
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def
  raise RuntimeError("MetaGraphDef associated with tag-set {0} could not be found in SavedModel".format(tag_set))

def yield_batch(iterable, batch_size):
  """Generator that yields batches of an iterator"""
  batch = []
  for item in iterable:
    batch.append(item)
    if len(batch) >= batch_size:
      yield batch
      batch = []
  if len(batch) > 0:
      yield batch

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

from tensorflow.contrib.saved_model.python.saved_model import reader, signature_def_utils
from tensorflow.python.client import session
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader
from tensorflowonspark import TFCluster

class TFEstimator(Estimator):
  train_fn = None
  args = None
  input_mode = None

  def __init__(self, train_fn, args, input_mode):
    self.train_fn = train_fn
    self.args = args
    self.input_mode = input_mode

  def _fit(self, dataset):
    self.args.mode = 'train'
    print("===== train args: {0}".format(self.args))
    sc = SparkContext.getOrCreate()
    cluster = TFCluster.run(sc, self.train_fn, self.args, self.args.cluster_size, self.args.num_ps, self.args.tensorboard, self.input_mode)
    cluster.train(dataset.rdd, self.args.epochs)
    cluster.shutdown()
    return TFModel(self.args)

class TFModel(Model):
  args = None

  def __init__(self, args):
    self.args = args

  def _transform(self, dataset):
    self.args.mode = 'inference'
    print("===== inference args: {0}".format(self.args))
    spark = SparkSession.builder.getOrCreate()
    rdd_out = dataset.rdd.mapPartitions(lambda it: _run_saved_model(it, self.args))
    return spark.createDataFrame(rdd_out, "string")

# Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py#L233
def _run_saved_model(iterator, args):
  print("===== loading meta_graph_def")
  meta_graph_def = get_meta_graph_def(args.export_dir, args.tag_set)
  inputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, args.signature_def_key).inputs
  print("inputs_tensor_info: {0}".format(inputs_tensor_info))
  outputs_tensor_info = signature_def_utils.get_signature_def_by_key(meta_graph_def, args.signature_def_key).outputs
  print("outputs_tensor_info: {0}".format(outputs_tensor_info))

  result = []
  with session.Session(graph=ops_lib.Graph()) as sess:
    print("===== loading saved_model")
    loader.load(sess, args.tag_set.split(','), args.export_dir)
    print("===== running saved_model")
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

def yield_batch(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def get_meta_graph_def(saved_model_dir, tag_set):
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def

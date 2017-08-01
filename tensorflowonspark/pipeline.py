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

import copy
import logging
import os
import subprocess

##### TensorFlowOnSpark Params

class HasBatchSize(Params):
  batch_size = Param(Params._dummy(), "batch_size", "Number of records per batch", typeConverter=TypeConverters.toInt)
  def __init__(self):
    super(HasBatchSize, self).__init__()
  def setBatchSize(self, value):
    return self._set(batch_size=value)
  def getBatchSize(self):
    return self.getOrDefault(self.batch_size)

class HasClusterSize(Params):
  cluster_size = Param(Params._dummy(), "cluster_size", "Number of nodes in the cluster", typeConverter=TypeConverters.toInt)
  def __init__(self):
    super(HasClusterSize, self).__init__()
  def setClusterSize(self, value):
    return self._set(cluster_size=value)
  def getClusterSize(self):
    return self.getOrDefault(self.cluster_size)

class HasEpochs(Params):
  epochs = Param(Params._dummy(), "epochs", "Number of epochs to train", typeConverter=TypeConverters.toInt)
  def __init__(self):
    super(HasEpochs, self).__init__()
  def setEpochs(self, value):
    return self._set(epochs=value)
  def getEpochs(self):
    return self.getOrDefault(self.epochs)

class HasInputTensor(Params):
  tensor_in = Param(Params._dummy(), "tensor_in", "Name of input tensor in signature def", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasInputTensor, self).__init__()
  def setInputTensor(self, value):
    return self._set(tensor_in=value)
  def getInputTensor(self):
    return self.getOrDefault(self.tensor_in)

class HasModelDir(Params):
  model_dir = Param(Params._dummy(), "model_dir", "Path to save/load model checkpoints", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasModelDir, self).__init__()
  def setModelDir(self, value):
    return self._set(model_dir=value)
  def getModelDir(self):
    return self.getOrDefault(self.model_dir)

class HasNumPS(Params):
  num_ps = Param(Params._dummy(), "num_ps", "Number of PS nodes in cluster", typeConverter=TypeConverters.toInt)
  def __init__(self):
    super(HasNumPS, self).__init__()
  def setNumPS(self, value):
    return self._set(num_ps=value)
  def getNumPS(self):
    return self.getOrDefault(self.num_ps)

class HasOutputTensor(Params):
  tensor_out = Param(Params._dummy(), "tensor_out", "Name of output tensor in signature def", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasOutputTensor, self).__init__()
  def setOutputTensor(self, value):
    return self._set(tensor_out=value)
  def getOutputTensor(self):
    return self.getOrDefault(self.tensor_out)

class HasRDMA(Params):
  rdma = Param(Params._dummy(), "rdma", "Use RDMA connection", typeConverter=TypeConverters.toBoolean)
  def __init__(self):
    super(HasRDMA, self).__init__()
  def setRDMA(self, value):
    return self._set(rdma=value)
  def getRDMA(self):
    return self.getOrDefault(self.rdma)

class HasSteps(Params):
  steps = Param(Params._dummy(), "steps", "Maximum number of steps to train", typeConverter=TypeConverters.toInt)
  def __init__(self):
    super(HasSteps, self).__init__()
  def setSteps(self, value):
    return self._set(steps=value)
  def getSteps(self):
    return self.getOrDefault(self.steps)

class HasTensorboard(Params):
  tensorboard = Param(Params._dummy(), "tensorboard", "Launch tensorboard process", typeConverter=TypeConverters.toBoolean)
  def __init__(self):
    super(HasTensorboard, self).__init__()
  def setTensorboard(self, value):
    return self._set(tensorboard=value)
  def getTensorboard(self):
    return self.getOrDefault(self.tensorboard)

##### SavedModelBuilder Params

class HasExportDir(Params):
  export_dir = Param(Params._dummy(), "export_dir", "Directory to export saved_model", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasExportDir, self).__init__()
  def setExportDir(self, value):
    return self._set(export_dir=value)
  def getExportDir(self):
    return self.getOrDefault(self.export_dir)

class HasMethodName(Params):
  method_name = Param(Params._dummy(), "method_name", "Method name for a saved_model signature", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasMethodName, self).__init__()
  def setMethodName(self, value):
    return self._set(method_name=value)
  def getMethodName(self):
    return self.getOrDefault(self.method_name)

class HasSignatureDefKey(Params):
  signature_def_key = Param(Params._dummy(), "signature_def_key", "Identifier for a specific saved_model signature", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasSignatureDefKey, self).__init__()
  def setSignatureDefKey(self, value):
    return self._set(signature_def_key=value)
  def getSignatureDefKey(self):
    return self.getOrDefault(self.signature_def_key)

class HasTagSet(Params):
  tag_set = Param(Params._dummy(), "tag_set", "Comma-delimited list of tags identifying a saved_model metagraph", typeConverter=TypeConverters.toString)
  def __init__(self):
    super(HasTagSet, self).__init__()
  def setTagSet(self, value):
    return self._set(tag_set=value)
  def getTagSet(self):
    return self.getOrDefault(self.tag_set)

class Namespace(object):
  """
  Utility class to convert dictionaries to Namespace-like objects
  Based on https://docs.python.org/dev/library/types.html#types.SimpleNamespace
  """
  def __init__(self, d):
    self.__dict__.update(d)
  def __repr__(self):
    keys = sorted(self.__dict__)
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}({})".format(type(self).__name__, ", ".join(items))
  def __eq__(self, other):
    return self.__dict__ == other.__dict__

class TFParams(Params):
  """Mix-in class to store args and merge params"""
  args = None
  def _merge_args_params(self):
    local_args = copy.copy(self.args)
    args_dict = vars(local_args)
    for p in self.params:
      args_dict[p.name] = self.getOrDefault(p.name)
    return local_args

class TFEstimator(Estimator, TFParams, HasInputCol, HasPredictionCol,
                  HasInputTensor, HasOutputTensor,
                  HasClusterSize, HasNumPS, HasRDMA, HasTensorboard, HasModelDir,
                  HasBatchSize, HasEpochs, HasSteps,
                  HasExportDir, HasMethodName, HasSignatureDefKey, HasTagSet):
  """Spark ML Pipeline Estimator which launches a TensorFlowOnSpark cluster for training"""

  train_fn = None

  def __init__(self, train_fn, tf_args):
    super(TFEstimator, self).__init__()
    self.train_fn = train_fn
    self.args = Namespace(tf_args) if isinstance(tf_args, dict) else tf_args
    self._setDefault(inputCol='images',
                    predictionCol='prediction',
                    tensor_in='input',
                    tensor_out='output',
                    cluster_size=1,
                    num_ps=0,
                    rdma=False,
                    tensorboard=False,
                    model_dir='tf_model',
                    batch_size=100,
                    epochs=1,
                    steps=1000,
                    export_dir='tf_export',
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    signature_def_key=tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                    tag_set=tf.saved_model.tag_constants.SERVING)

  def _fit(self, dataset):
    sc = SparkContext.getOrCreate()

    logging.info("===== 1. train args: {0}".format(self.args))
    logging.info("===== 2. train params: {0}".format(self._paramMap))
    local_args = self._merge_args_params()
    logging.info("===== 3. train args + params: {0}".format(local_args))

    cluster = TFCluster.run(sc, self.train_fn, local_args, local_args.cluster_size, local_args.num_ps, local_args.tensorboard, TFCluster.InputMode.SPARK)
    cluster.train(dataset.rdd, local_args.epochs)
    cluster.shutdown()
    return self._copyValues(TFModel(self.args))

class TFModel(Model, TFParams, HasInputCol, HasPredictionCol,
              HasInputTensor, HasOutputTensor,
              HasBatchSize,
              HasExportDir, HasMethodName, HasSignatureDefKey, HasTagSet):
  """Spark ML Pipeline Model which runs a TensorFlow SavedModel stored on disk."""

  def __init__(self, args):
    super(TFModel, self).__init__()
    self.args = args

  def _transform(self, dataset):
    spark = SparkSession.builder.getOrCreate()

    logging.info("===== 1. inference args: {0}".format(self.args))
    logging.info("===== 2. inference params: {0}".format(self._paramMap))
    local_args = self._merge_args_params()
    logging.info("===== 2. inference args + params: {0}".format(local_args))

    rdd_out = dataset.rdd.mapPartitions(lambda it: _run_saved_model(it, local_args))
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

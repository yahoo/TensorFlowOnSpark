# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module extends the TensorFlowOnSpark API to support Spark ML Pipelines.

It provides a TFEstimator class to fit a TFModel using TensorFlow.  The TFEstimator will actually spawn a TensorFlowOnSpark cluster
to conduct distributed training, but due to architectural limitations, the TFModel will only run single-node TensorFlow instances
when inferencing on the executors.  The executors will run in parallel, but the TensorFlow model must fit in the memory
of each executor.

There is also an option to provide a separate "export" function, which allows users to export a different graph for inferencing vs. training.
This is useful when the training graph uses InputMode.TENSORFLOW with queue_runners, but the inferencing graph needs placeholders.
And this is especially useful for exporting saved_models for TensorFlow Serving.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import Estimator, Model
from pyspark.sql import Row, SparkSession

import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader, signature_def_utils
from tensorflow.python.saved_model import loader
from . import TFCluster, gpu_info, dfutil

import argparse
import copy
import logging
import os
import subprocess
import sys


# TensorFlowOnSpark Params

class TFTypeConverters(object):
  """Custom DataFrame TypeConverter for dictionary types (since this is not provided by Spark core)."""
  @staticmethod
  def toDict(value):
    if type(value) == dict:
      return value
    else:
      raise TypeError("Could not convert %s to OrderedDict" % value)


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


class HasInputMapping(Params):
  input_mapping = Param(Params._dummy(), "input_mapping", "Mapping of input DataFrame column to input tensor", typeConverter=TFTypeConverters.toDict)

  def __init__(self):
    super(HasInputMapping, self).__init__()

  def setInputMapping(self, value):
    return self._set(input_mapping=value)

  def getInputMapping(self):
    return self.getOrDefault(self.input_mapping)


class HasInputMode(Params):
  input_mode = Param(Params._dummy(), "input_mode", "Input data feeding mode (0=TENSORFLOW, 1=SPARK)", typeConverter=TypeConverters.toInt)

  def __init__(self):
    super(HasInputMode, self).__init__()

  def setInputMode(self, value):
    return self._set(input_mode=value)

  def getInputMode(self):
    return self.getOrDefault(self.input_mode)


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
  driver_ps_nodes = Param(Params._dummy(), "driver_ps_nodes", "Run PS nodes on driver locally", typeConverter=TypeConverters.toBoolean)

  def __init__(self):
    super(HasNumPS, self).__init__()

  def setNumPS(self, value):
    return self._set(num_ps=value)

  def getNumPS(self):
    return self.getOrDefault(self.num_ps)

  def setDriverPSNodes(self, value):
    return self._set(driver_ps_nodes=value)

  def getDriverPSNodes(self):
    return self.getOrDefault(self.driver_ps_nodes)


class HasOutputMapping(Params):
  output_mapping = Param(Params._dummy(), "output_mapping", "Mapping of output tensor to output DataFrame column", typeConverter=TFTypeConverters.toDict)

  def __init__(self):
    super(HasOutputMapping, self).__init__()

  def setOutputMapping(self, value):
    return self._set(output_mapping=value)

  def getOutputMapping(self):
    return self.getOrDefault(self.output_mapping)


class HasProtocol(Params):
  protocol = Param(Params._dummy(), "protocol", "Network protocol for Tensorflow (grpc|rdma)", typeConverter=TypeConverters.toString)

  def __init__(self):
    super(HasProtocol, self).__init__()

  def setProtocol(self, value):
    return self._set(protocol=value)

  def getProtocol(self):
    return self.getOrDefault(self.protocol)


class HasReaders(Params):
  readers = Param(Params._dummy(), "readers", "number of reader/enqueue threads", typeConverter=TypeConverters.toInt)

  def __init__(self):
    super(HasReaders, self).__init__()

  def setReaders(self, value):
    return self._set(readers=value)

  def getReaders(self):
    return self.getOrDefault(self.readers)


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


class HasTFRecordDir(Params):
  tfrecord_dir = Param(Params._dummy(), "tfrecord_dir", "Path to temporarily export a DataFrame as TFRecords (for InputMode.TENSORFLOW apps)", typeConverter=TypeConverters.toString)

  def __init__(self):
    super(HasTFRecordDir, self).__init__()

  def setTFRecordDir(self, value):
    return self._set(tfrecord_dir=value)

  def getTFRecordDir(self):
    return self.getOrDefault(self.tfrecord_dir)


# SavedModelBuilder Params

class HasExportDir(Params):
  export_dir = Param(Params._dummy(), "export_dir", "Directory to export saved_model", typeConverter=TypeConverters.toString)

  def __init__(self):
    super(HasExportDir, self).__init__()

  def setExportDir(self, value):
    return self._set(export_dir=value)

  def getExportDir(self):
    return self.getOrDefault(self.export_dir)


class HasSignatureDefKey(Params):
  signature_def_key = Param(Params._dummy(), "signature_def_key", "Identifier for a specific saved_model signature", typeConverter=TypeConverters.toString)

  def __init__(self):
    super(HasSignatureDefKey, self).__init__()
    self._setDefault(signature_def_key=None)

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
  Utility class to convert dictionaries to Namespace-like objects.

  Based on https://docs.python.org/dev/library/types.html#types.SimpleNamespace
  """
  argv = None

  def __init__(self, d):
    if isinstance(d, list):
      self.argv = d
    elif isinstance(d, dict):
      self.__dict__.update(d)
    elif isinstance(d, argparse.Namespace):
      self.__dict__.update(vars(d))
    elif isinstance(d, Namespace):
      self.__dict__.update(d.__dict__)
    else:
      raise Exception("Unsupported Namespace args: {}".format(d))

  def __iter__(self):
    if self.argv:
      for item in self.argv:
        yield item
    else:
      for key in self.__dict__.keys():
        yield key

  def __repr__(self):
    if self.argv:
      return "{}".format(self.argv)
    else:
      keys = sorted(self.__dict__)
      items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
      return "{}({})".format(type(self).__name__, ", ".join(items))

  def __eq__(self, other):
    if self.argv:
      return self.argv == other
    else:
      return self.__dict__ == other.__dict__


class TFParams(Params):
  """Mix-in class to store namespace-style args and merge w/ SparkML-style params."""
  args = None

  def merge_args_params(self):
    local_args = copy.copy(self.args)                 # make a local copy of args
    args_dict = vars(local_args)                      # get dictionary view
    for p in self.params:
      args_dict[p.name] = self.getOrDefault(p.name)   # update with params
    return local_args


class TFEstimator(Estimator, TFParams, HasInputMapping,
                  HasClusterSize, HasNumPS, HasInputMode, HasProtocol, HasTensorboard, HasModelDir, HasExportDir, HasTFRecordDir,
                  HasBatchSize, HasEpochs, HasReaders, HasSteps):
  """Spark ML Estimator which launches a TensorFlowOnSpark cluster for distributed training.

  The columns of the DataFrame passed to the ``fit()`` method will be mapped to TensorFlow tensors according to the ``setInputMapping()`` method.

  If an ``export_fn`` was provided to the constructor, it will be run on a single executor immediately after the distributed training has completed.
  This allows users to export a TensorFlow saved_model with a different execution graph for inferencing, e.g. replacing an input graph of
  TFReaders and QueueRunners with Placeholders.

  For InputMode.TENSORFLOW, the input DataFrame will be exported as TFRecords to a temporary location specified by the ``tfrecord_dir``.
  The TensorFlow application will then be expected to read directly from this location during training.  However, if the input DataFrame was
  produced by the ``dfutil.loadTFRecords()`` method, i.e. originated from TFRecords on disk, then the `tfrecord_dir` will be set to the
  original source location of the TFRecords with the additional export step.

  Args:
    :train_fn: TensorFlow "main" function for training.
    :tf_args: Arguments specific to the TensorFlow "main" function.
    :export_fn: TensorFlow function for exporting a saved_model.
  """

  train_fn = None
  export_fn = None

  def __init__(self, train_fn, tf_args, export_fn=None):
    super(TFEstimator, self).__init__()
    self.train_fn = train_fn
    self.export_fn = export_fn
    self.args = Namespace(tf_args)
    self._setDefault(input_mapping={},
                     cluster_size=1,
                     num_ps=0,
                     driver_ps_nodes=False,
                     input_mode=TFCluster.InputMode.SPARK,
                     protocol='grpc',
                     tensorboard=False,
                     model_dir=None,
                     export_dir=None,
                     tfrecord_dir=None,
                     batch_size=100,
                     epochs=1,
                     readers=1,
                     steps=1000)

  def _fit(self, dataset):
    """Trains a TensorFlow model and returns a TFModel instance with the same args/params pointing to a checkpoint or saved_model on disk.

    Args:
      :dataset: A Spark DataFrame with columns that will be mapped to TensorFlow tensors.

    Returns:
      A TFModel representing the trained model, backed on disk by a TensorFlow checkpoint or saved_model.
    """
    sc = SparkContext.getOrCreate()

    logging.info("===== 1. train args: {0}".format(self.args))
    logging.info("===== 2. train params: {0}".format(self._paramMap))
    local_args = self.merge_args_params()
    logging.info("===== 3. train args + params: {0}".format(local_args))

    if local_args.input_mode == TFCluster.InputMode.TENSORFLOW:
      if dfutil.isLoadedDF(dataset):
        # if just a DataFrame loaded from tfrecords, just point to original source path
        logging.info("Loaded DataFrame of TFRecord.")
        local_args.tfrecord_dir = dfutil.loadedDF[dataset]
      else:
        # otherwise, save as tfrecords and point to save path
        assert local_args.tfrecord_dir, "Please specify --tfrecord_dir to export DataFrame to TFRecord."
        if self.getInputMapping():
          # if input mapping provided, filter only required columns before exporting
          dataset = dataset.select(self.getInputMapping().keys())
        logging.info("Exporting DataFrame {} as TFRecord to: {}".format(dataset.dtypes, local_args.tfrecord_dir))
        dfutil.saveAsTFRecords(dataset, local_args.tfrecord_dir)
        logging.info("Done saving")

    tf_args = self.args.argv if self.args.argv else local_args
    cluster = TFCluster.run(sc, self.train_fn, tf_args, local_args.cluster_size, local_args.num_ps,
                            local_args.tensorboard, local_args.input_mode, driver_ps_nodes=local_args.driver_ps_nodes)
    if local_args.input_mode == TFCluster.InputMode.SPARK:
      # feed data, using a deterministic order for input columns (lexicographic by key)
      input_cols = sorted(self.getInputMapping().keys())
      cluster.train(dataset.select(input_cols).rdd, local_args.epochs)
    cluster.shutdown()

    # Run export function, if provided
    if self.export_fn:
      assert local_args.export_dir, "Export function requires --export_dir to be set"
      logging.info("Exporting saved_model (via export_fn) to: {}".format(local_args.export_dir))

      def _export(iterator, fn, args):
        single_node_env(args)
        fn(args)

      # Run on a single exeucutor
      sc.parallelize([1], 1).foreachPartition(lambda it: _export(it, self.export_fn, tf_args))

    return self._copyValues(TFModel(self.args))


class TFModel(Model, TFParams,
              HasInputMapping, HasOutputMapping,
              HasBatchSize,
              HasModelDir, HasExportDir, HasSignatureDefKey, HasTagSet):
  """Spark ML Model backed by a TensorFlow model checkpoint/saved_model on disk.

  During ``transform()``, each executor will run an independent, single-node instance of TensorFlow in parallel, so the model must fit in memory.
  The model/session will be loaded/initialized just once for each Spark Python worker, and the session will be cached for
  subsequent tasks/partitions to avoid re-loading the model for each partition.

  Args:
    :tf_args: Dictionary of arguments specific to TensorFlow "main" function.
  """

  def __init__(self, tf_args):
    super(TFModel, self).__init__()
    self.args = Namespace(tf_args)
    self._setDefault(input_mapping={},
                     output_mapping={},
                     batch_size=100,
                     model_dir=None,
                     export_dir=None,
                     signature_def_key=None,
                     tag_set=None)

  def _transform(self, dataset):
    """Transforms the input DataFrame by applying the _run_model() mapPartitions function.

    Args:
      :dataset: A Spark DataFrame for TensorFlow inferencing.
    """
    spark = SparkSession.builder.getOrCreate()

    # set a deterministic order for input/output columns (lexicographic by key)
    input_cols = [col for col, tensor in sorted(self.getInputMapping().items())]      # input col => input tensor
    output_cols = [col for tensor, col in sorted(self.getOutputMapping().items())]    # output tensor => output col

    # run single-node inferencing on each executor
    logging.info("input_cols: {}".format(input_cols))
    logging.info("output_cols: {}".format(output_cols))

    # merge args + params
    logging.info("===== 1. inference args: {0}".format(self.args))
    logging.info("===== 2. inference params: {0}".format(self._paramMap))
    local_args = self.merge_args_params()
    logging.info("===== 3. inference args + params: {0}".format(local_args))

    tf_args = self.args.argv if self.args.argv else local_args
    rdd_out = dataset.select(input_cols).rdd.mapPartitions(lambda it: _run_model(it, local_args, tf_args))

    # convert to a DataFrame-friendly format
    rows_out = rdd_out.map(lambda x: Row(*x))
    return spark.createDataFrame(rows_out, output_cols)


# global to each python worker process on the executors
global_sess = None            # tf.Session cache
global_args = None            # args provided to the _run_model() method.  Any change will invalidate the global_sess cache.


def _run_model(iterator, args, tf_args):
  """mapPartitions function to run single-node inferencing from a checkpoint/saved_model, using the model's input/output mappings.

  Args:
    :iterator: input RDD partition iterator.
    :args: arguments for TFModel, in argparse format
    :tf_args: arguments for TensorFlow inferencing code, in argparse or ARGV format.

  Returns:
    An iterator of result data.
  """
  single_node_env(tf_args)

  logging.info("===== input_mapping: {}".format(args.input_mapping))
  logging.info("===== output_mapping: {}".format(args.output_mapping))
  input_tensor_names = [tensor for col, tensor in sorted(args.input_mapping.items())]
  output_tensor_names = [tensor for tensor, col in sorted(args.output_mapping.items())]

  # if using a signature_def_key, get input/output tensor info from the requested signature
  if args.signature_def_key:
    assert args.export_dir, "Inferencing with signature_def_key requires --export_dir argument"
    logging.info("===== loading meta_graph_def for tag_set ({0}) from saved_model: {1}".format(args.tag_set, args.export_dir))
    meta_graph_def = get_meta_graph_def(args.export_dir, args.tag_set)
    signature = signature_def_utils.get_signature_def_by_key(meta_graph_def, args.signature_def_key)
    logging.debug("signature: {}".format(signature))
    inputs_tensor_info = signature.inputs
    logging.debug("inputs_tensor_info: {0}".format(inputs_tensor_info))
    outputs_tensor_info = signature.outputs
    logging.debug("outputs_tensor_info: {0}".format(outputs_tensor_info))

  result = []

  global global_sess, global_args
  if global_sess and global_args == args:
    # if graph/session already loaded/started (and using same args), just reuse it
    sess = global_sess
  else:
    # otherwise, create new session and load graph from disk
    tf.reset_default_graph()
    sess = tf.Session(graph=tf.get_default_graph())
    if args.export_dir:
      assert args.tag_set, "Inferencing from a saved_model requires --tag_set"
      # load graph from a saved_model
      logging.info("===== restoring from saved_model: {}".format(args.export_dir))
      loader.load(sess, args.tag_set.split(','), args.export_dir)
    elif args.model_dir:
      # load graph from a checkpoint
      ckpt = tf.train.latest_checkpoint(args.model_dir)
      assert ckpt, "Invalid model checkpoint path: {}".format(args.model_dir)
      logging.info("===== restoring from checkpoint: {}".format(ckpt + ".meta"))
      saver = tf.train.import_meta_graph(ckpt + ".meta", clear_devices=True)
      saver.restore(sess, ckpt)
    else:
      raise Exception("Inferencing requires either --model_dir or --export_dir argument")
    global_sess = sess
    global_args = args

  # get list of input/output tensors (by name)
  if args.signature_def_key:
    input_tensors = [inputs_tensor_info[t].name for t in input_tensor_names]
    output_tensors = [outputs_tensor_info[output_tensor_names[0]].name]
  else:
    input_tensors = [t + ':0' for t in input_tensor_names]
    output_tensors = [t + ':0' for t in output_tensor_names]

  logging.info("input_tensors: {0}".format(input_tensors))
  logging.info("output_tensors: {0}".format(output_tensors))

  # feed data in batches and return output tensors
  for tensors in yield_batch(iterator, args.batch_size, len(input_tensor_names)):
    inputs_feed_dict = {}
    for i in range(len(input_tensors)):
      inputs_feed_dict[input_tensors[i]] = tensors[i]

    outputs = sess.run(output_tensors, feed_dict=inputs_feed_dict)
    lengths = [len(output) for output in outputs]
    input_size = len(tensors[0])
    assert all([length == input_size for length in lengths]), "Output array sizes {} must match input size: {}".format(lengths, input_size)
    python_outputs = [output.tolist() for output in outputs]      # convert from numpy to standard python types
    result.extend(zip(*python_outputs))                           # convert to an array of tuples of "output columns"

  return result


def single_node_env(args):
  """Sets up environment for a single-node TF session.

  Args:
    :args: command line arguments as either argparse args or argv list
  """
  if isinstance(args, list):
      sys.argv = args
  elif args.argv:
      sys.argv = args.argv

  # ensure expanded CLASSPATH w/o glob characters (required for Spark 2.1 + JNI)
  if 'HADOOP_PREFIX' in os.environ and 'TFOS_CLASSPATH_UPDATED' not in os.environ:
      classpath = os.environ['CLASSPATH']
      hadoop_path = os.path.join(os.environ['HADOOP_PREFIX'], 'bin', 'hadoop')
      hadoop_classpath = subprocess.check_output([hadoop_path, 'classpath', '--glob']).decode()
      logging.debug("CLASSPATH: {0}".format(hadoop_classpath))
      os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath
      os.environ['TFOS_CLASSPATH_UPDATED'] = '1'

  # reserve GPU, if requested
  if tf.test.is_built_with_cuda():
    # GPU
    num_gpus = args.num_gpus if 'num_gpus' in args else 1
    gpus_to_use = gpu_info.get_gpus(num_gpus)
    logging.info("Using gpu(s): {0}".format(gpus_to_use))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    # Note: if there is a GPU conflict (CUDA_ERROR_INVALID_DEVICE), the entire task will fail and retry.
  else:
    # CPU
    logging.info("Using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_meta_graph_def(saved_model_dir, tag_set):
  """Utility function to read a meta_graph_def from disk.

  From `saved_model_cli.py <https://github.com/tensorflow/tensorflow/blob/8e0e8d41a3a8f2d4a6100c2ea1dc9d6c6c4ad382/tensorflow/python/tools/saved_model_cli.py#L186>`_

  Args:
    :saved_model_dir: path to saved_model.
    :tag_set: list of string tags identifying the TensorFlow graph within the saved_model.

  Returns:
    A TensorFlow meta_graph_def, or raises an Exception otherwise.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def
  raise RuntimeError("MetaGraphDef associated with tag-set {0} could not be found in SavedModel".format(tag_set))


def yield_batch(iterable, batch_size, num_tensors=1):
  """Generator that yields batches of a DataFrame iterator.

  Args:
    :iterable: Spark partition iterator.
    :batch_size: number of items to retrieve per invocation.
    :num_tensors: number of tensors (columns) expected in each item.

  Returns:
    An array of ``num_tensors`` arrays, each of length `batch_size`
  """
  tensors = [[] for i in range(num_tensors)]
  for item in iterable:
    if item is None:
      break
    for i in range(num_tensors):
      tmp = str(item[i]) if type(item[i]) is bytearray else item[i]
      tensors[i].append(tmp)
    if len(tensors[0]) >= batch_size:
      yield tensors
      tensors = [[] for i in range(num_tensors)]
  if len(tensors[0]) > 0:
      yield tensors

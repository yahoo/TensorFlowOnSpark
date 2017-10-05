# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""
This module provides TensorFlow helper functions for allocating GPUs and interacting with the Spark executor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import getpass
import logging
import os
import time
from six.moves.queue import Empty
from . import marker

def hdfs_path(ctx, path):
  """Convenience function to create a Tensorflow-compatible absolute HDFS path from relative paths

  Args:
    ctx: TFNodeContext containing the metadata specific to this node in the cluster.
    path: path to convert

  Returns:
    An absolute path prefixed with the correct filesystem scheme.
  """
  if path.startswith("hdfs://") or path.startswith("viewfs://") or path.startswith("file://"):
    # absolute path w/ scheme, just return as-is
    return path
  elif path.startswith("/"):
    # absolute path w/o scheme, just prepend w/ defaultFS
    return ctx.defaultFS + path
  else:
    # relative path, prepend defaultSF + standard working dir
    if ctx.defaultFS.startswith("hdfs://") or ctx.defaultFS.startswith("viewfs://"):
      return "{0}/user/{1}/{2}".format(ctx.defaultFS, getpass.getuser(), path)
    elif ctx.defaultFS.startswith("file://"):
      return "{0}/{1}/{2}".format(ctx.defaultFS, ctx.working_dir[1:], path)
    else:
      logging.warn("Unknown scheme {0} with relative path: {1}".format(ctx.defaultFS, path))
      return "{0}/{1}".format(ctx.defaultFS, path)

def start_cluster_server(ctx, num_gpus=1, rdma=False):
  """Function that wraps the creation of TensorFlow tf.train.Server for a node in a distributed TensorFlow cluster.

  This is intended to be invoked from within the TF map_fun, replacing explicit code to instantiate `tf.train.ClusterSpec`
  and `tf.train.Server` objects.

  Args:
    ctx: TFNodeContext containing the metadata specific to this node in the cluster.
    num_gpu: number of GPUs desired
    rdma: boolean indicating if RDMA 'iverbs' should be used for cluster communications.

  Returns:
    A tuple of (cluster_spec, server)
  """
  import tensorflow as tf
  from . import gpu_info

  logging.info("{0}: ======== {1}:{2} ========".format(ctx.worker_num, ctx.job_name, ctx.task_index))
  cluster_spec = ctx.cluster_spec
  logging.info("{0}: Cluster spec: {1}".format(ctx.worker_num, cluster_spec))

  if tf.test.is_built_with_cuda():
    # GPU
    gpu_initialized = False
    while not gpu_initialized:
      try:
        # override PS jobs to only reserve one GPU
        if ctx.job_name == 'ps':
          num_gpus = 1

        # Find a free gpu(s) to use
        gpus_to_use = gpu_info.get_gpus(num_gpus)
        gpu_prompt = "GPU" if num_gpus == 1 else "GPUs"
        logging.info("{0}: Using {1}: {2}".format(ctx.worker_num, gpu_prompt, gpus_to_use))

        # Set GPU device to use for TensorFlow
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec(cluster_spec)

        # Create and start a server for the local task.
        if rdma:
          server = tf.train.Server(cluster, ctx.job_name, ctx.task_index, protocol="grpc_rdma")
        else:
          server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)
        gpu_initialized = True
      except Exception as e:
        print(e)
        logging.error("{0}: Failed to allocate GPU, trying again...".format(ctx.worker_num))
        time.sleep(10)
  else:
    # CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logging.info("{0}: Using CPU".format(ctx.worker_num))

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)

  return (cluster, server)

def next_batch(mgr, batch_size, qname='input'):
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")

def export_saved_model(sess, export_dir, tag_set, signatures):
  """Convenience function to export a saved_model using provided arguments

  The caller specifies the saved_model signatures in a simplified python dictionary form, as follows:

    signatures = {
      'signature_def_key': {
        'inputs': { 'input_tensor_alias': input_tensor_name },
        'outputs': { 'output_tensor_alias': output_tensor_name },
        'method_name': 'method'
      }
    }

  And this function will generate the `signature_def_map` and export the saved_model.

  Args:
    sess: a tf.Session instance
    export_dir: path to save exported saved_model
    tag_set: string tag_set to identify the exported graph
    signatures: simplified dictionary representation of a TensorFlow `signature_def_map`

  Returns:
    A saved_model exported to disk at `export_dir`.
  """
  import tensorflow as tf
  g = sess.graph
  g._unsafe_unfinalize()           # https://github.com/tensorflow/serving/issues/363
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  logging.info("===== signatures: {}".format(signatures))
  signature_def_map = {}
  for key, sig in signatures.items():
    signature_def_map[key] = tf.saved_model.signature_def_utils.build_signature_def(
              inputs={ name:tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in sig['inputs'].items() },
              outputs={ name:tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in sig['outputs'].items() },
              method_name=sig['method_name'] if 'method_name' in sig else key)
  logging.info("===== signature_def_map: {}".format(signature_def_map))
  builder.add_meta_graph_and_variables(sess,
              tag_set.split(','),
              signature_def_map=signature_def_map,
              clear_devices=True)
  g.finalize()
  builder.save()

def batch_results(mgr, results, qname='output'):
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")

def terminate(mgr, qname='input'):
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")

class DataFeed(object):
  """Manages the InputMode.SPARK data feeding process from the perspective of the TensorFlow application"""
  def __init__(self, mgr, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):
    """DataFeed constructor

    Args:
      mgr: the TFManager associated with this Python worker process
      train_mode: boolean indicating if the data feed is expecting an output response (e.g. inferencing)
      qname_in: INTERNAL_USE
      qname_out: INTERNAL_USE
      input_mapping: For Spark ML Pipelines ONLY.  Specifies the mapping of input DataFrame colums to input tensors
    """
    self.mgr = mgr
    self.train_mode = train_mode
    self.qname_in = qname_in
    self.qname_out = qname_out
    self.done_feeding = False
    self.input_tensors = [ tensor for col, tensor in sorted(input_mapping.items()) ] if input_mapping is not None else None

  def next_batch(self, batch_size):
    """Gets a batch of items from the input RDD as either an array of items or a dict of tensors (depending on the input_mapping).

    If multiple tensors are provided per row, e.g. tuple of (tensor1, tensor2, ..., tensorN) and:
    - no input_mapping is provided, this will just return an array of tuples, and the caller is responsible for separating the tensors.
    - an input_mapping is provided, this will return a dictionary of tensors.

    Note: if the end of the data is reached, this may return with fewer than `batch_size` items.

    Args:
      batch_size: number of items to return.

    Returns:
      A batch of items or an array of tensors.
    """
    logging.debug("next_batch() invoked")
    queue = self.mgr.get_queue(self.qname_in)
    tensors = [] if self.input_tensors is None else { tensor:[] for tensor in self.input_tensors }
    count = 0
    while count < batch_size:
      item = queue.get(block=True)
      if item is None:
        # End of Feed
        logging.info("next_batch() got None")
        queue.task_done()
        self.done_feeding = True
        break
      elif type(item) is marker.EndPartition:
        # End of Partition
        logging.info("next_batch() got EndPartition")
        queue.task_done()
        if not self.train_mode and count > 0:
          break
      else:
        # Normal item
        if self.input_tensors is None:
          tensors.append(item)
        else:
          for i in range(len(self.input_tensors)):
            tensors[self.input_tensors[i]].append(item[i])
        count += 1
        queue.task_done()
    logging.debug("next_batch() returning {0} items".format(count))
    return tensors

  def should_stop(self):
    """Check if the feed process was told to stop."""
    return self.done_feeding

  def batch_results(self, results):
    """Push a batch of output results back to Spark.

    Note: this currently expects a one-to-one mapping of input data to output data.
    """
    logging.debug("batch_results() invoked")
    queue = self.mgr.get_queue(self.qname_out)
    for item in results:
      queue.put(item, block=True)
    logging.debug("batch_results() returning data")

  def terminate(self):
    """Terminate data feeding early

    Since TensorFlow applications can often terminate on conditions unrelated to the training data (e.g. steps, accuracy, etc),
    this method signals the data feeding process to drop any further incoming data.  Note that Spark itself does not have a mechanism
    to terminate an RDD operation early, so the extra partitions will still be sent to the executors (but will be ignored).
    """
    logging.info("terminate() invoked")
    self.mgr.set('state', 'terminating')

    # drop remaining items in the queue
    queue = self.mgr.get_queue(self.qname_in)
    count = 0
    done = False
    while not done:
      try:
        queue.get(block=True, timeout=5)
        queue.task_done()
        count += 1
      except Empty:
        logging.info("dropped {0} items from queue".format(count))
        done = True


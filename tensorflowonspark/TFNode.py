# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module provides helper functions for the TensorFlow application.

Primarily, these functions help with:

* starting the TensorFlow ``tf.train.Server`` for the node (allocating GPUs as desired, and determining the node's role in the cluster).
* managing input/output data for *InputMode.SPARK*.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import getpass
import logging
from six.moves.queue import Empty
from . import marker

logger = logging.getLogger(__name__)


def hdfs_path(ctx, path):
  """Convenience function to create a Tensorflow-compatible absolute HDFS path from relative paths

  Args:
    :ctx: TFNodeContext containing the metadata specific to this node in the cluster.
    :path: path to convert

  Returns:
    An absolute path prefixed with the correct filesystem scheme.
  """
  #  All Hadoop-Compatible File System Schemes (as of Hadoop 3.0.x):
  HADOOP_SCHEMES = ['adl://',
                    'file://',
                    'hdfs://',
                    'oss://',
                    's3://',
                    's3a://',
                    's3n://',
                    'swift://',
                    'viewfs://',
                    'wasb://']
  if (any(path.startswith(scheme) for scheme in HADOOP_SCHEMES)):
    # absolute path w/ scheme, just return as-is
    return path
  elif path.startswith("/"):
    # absolute path w/o scheme, just prepend w/ defaultFS
    return ctx.defaultFS + path
  else:
    # relative path, prepend defaultFS + standard working dir
    if ctx.defaultFS.startswith("hdfs://") or ctx.defaultFS.startswith("viewfs://"):
      return "{0}/user/{1}/{2}".format(ctx.defaultFS, getpass.getuser(), path)
    elif ctx.defaultFS.startswith("file://"):
      return "{0}/{1}/{2}".format(ctx.defaultFS, ctx.working_dir[1:], path)
    else:
      logger.warn("Unknown scheme {0} with relative path: {1}".format(ctx.defaultFS, path))
      return "{0}/{1}".format(ctx.defaultFS, path)


def start_cluster_server(ctx, num_gpus=1, rdma=False):
  """*DEPRECATED*.  Use higher-level APIs like `tf.keras` or `tf.estimator`"""
  raise Exception("DEPRECATED: Use higher-level APIs like `tf.keras` or `tf.estimator`")


def next_batch(mgr, batch_size, qname='input'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")


def export_saved_model(sess, export_dir, tag_set, signatures):
  """*DEPRECATED*. Use TF provided APIs instead."""
  raise Exception("DEPRECATED: Use TF provided APIs instead.")


def batch_results(mgr, results, qname='output'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")


def terminate(mgr, qname='input'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")


class DataFeed(object):
  """This class manages the *InputMode.SPARK* data feeding process from the perspective of the TensorFlow application.

  Args:
    :mgr: TFManager instance associated with this Python worker process.
    :train_mode: boolean indicating if the data feed is expecting an output response (e.g. inferencing).
    :qname_in: *INTERNAL_USE*
    :qname_out: *INTERNAL_USE*
    :input_mapping: *For Spark ML Pipelines only*.  Dictionary of input DataFrame columns to input TensorFlow tensors.
  """
  def __init__(self, mgr, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):

    self.mgr = mgr
    self.train_mode = train_mode
    self.qname_in = qname_in
    self.qname_out = qname_out
    self.done_feeding = False
    self.input_tensors = [tensor for col, tensor in sorted(input_mapping.items())] if input_mapping is not None else None

  def next_batch(self, batch_size):
    """Gets a batch of items from the input RDD.

    If multiple tensors are provided per row in the input RDD, e.g. tuple of (tensor1, tensor2, ..., tensorN) and:

    * no ``input_mapping`` was provided to the DataFeed constructor, this will return an array of ``batch_size`` tuples,
      and the caller is responsible for separating the tensors.
    * an ``input_mapping`` was provided to the DataFeed constructor, this will return a dictionary of N tensors,
      with tensor names as keys and arrays of length ``batch_size`` as values.

    Note: if the end of the data is reached, this may return with fewer than ``batch_size`` items.

    Args:
      :batch_size: number of items to retrieve.

    Returns:
      A batch of items or a dictionary of tensors.
    """
    logger.debug("next_batch() invoked")
    queue = self.mgr.get_queue(self.qname_in)
    tensors = [] if self.input_tensors is None else {tensor: [] for tensor in self.input_tensors}
    count = 0
    while count < batch_size:
      item = queue.get(block=True)
      if item is None:
        # End of Feed
        logger.info("next_batch() got None")
        queue.task_done()
        self.done_feeding = True
        break
      elif type(item) is marker.EndPartition:
        # End of Partition
        logger.info("next_batch() got EndPartition")
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
    logger.debug("next_batch() returning {0} items".format(count))
    return tensors

  def should_stop(self):
    """Check if the feed process was told to stop (by a call to ``terminate``)."""
    return self.done_feeding

  def batch_results(self, results):
    """Push a batch of output results to the Spark output RDD of ``TFCluster.inference()``.

    Note: this currently expects a one-to-one mapping of input to output data, so the length of the ``results`` array should match the length of
    the previously retrieved batch of input data.

    Args:
      :results: array of output data for the equivalent batch of input data.
    """
    logger.debug("batch_results() invoked")
    queue = self.mgr.get_queue(self.qname_out)
    for item in results:
      queue.put(item, block=True)
    logger.debug("batch_results() returning data")

  def terminate(self):
    """Terminate data feeding early.

    Since TensorFlow applications can often terminate on conditions unrelated to the training data (e.g. steps, accuracy, etc),
    this method signals the data feeding process to ignore any further incoming data.  Note that Spark itself does not have a mechanism
    to terminate an RDD operation early, so the extra partitions will still be sent to the executors (but will be ignored).  Because
    of this, you should size your input data accordingly to avoid excessive overhead.
    """
    logger.info("terminate() invoked")
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
        logger.info("dropped {0} items from queue".format(count))
        done = True

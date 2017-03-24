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
from six.moves.queue import Queue, Empty

def hdfs_path(ctx, path):
  """Convenience function to create a Tensorflow-compatible absolute HDFS path from relative paths"""
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
  """
  Wraps creation of TensorFlow Server in a distributed cluster.  This is intended to be invoked from the TF map_fun.
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
        gpus_to_use = gpu_info.get_gpus(num_gpus)[:num_gpus]
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
    os.environ['CUDA_VISIBLE-DEVICES'] = ''
    logging.info("{0}: Using CPU".format(ctx.worker_num))

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)

  return (cluster, server)


class TFNode(object):
    def __init__(self, mgr, batch_size, train_mode, qname_in='input', qname_out='output'):
        self.mgr = mgr
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.qname_in = qname_in
        self.qname_out = qname_out
        self.state = 0
        # 0: normal state
        # 1: Partition data processing is completed, if the received single None(len(bx)==0),
        # indicate that inference task is finished

    def next_batch(self):
        """
        Invoked from the user-supplied TensorFlow main function, which should have been launched as a background thread in the start() method
        with a multiprocessing.Manager as an argument.  This Manager and a unique queue name must be supplied to this function.
        """
        logging.debug("next_batch() invoked")
        queue = self.mgr.get_queue(self.qname_in)
        batch = []
        while len(batch) < self.batch_size:
            item = queue.get(block=True)
            if item is None:
                logging.info("next_batch() got None")
                queue.task_done()
                break
            else:
                # logging.info("next_batch() got {0}".format(item))
                batch.append(item)
                queue.task_done()
        self.__update_state(len(batch))
        logging.debug("next_batch() returning data")
        return batch

    def __update_state(self, size):
        if self.train_mode:
            if size < self.batch_size:
                self.state = 2
        else:
            if size == self.batch_size:
                self.state = 0
            elif size > 0:
                self.state = 1
            elif self.state == 0:
                self.state = 1
            else:
                self.state = 2

    def should_stop(self):
        """Check if the feed process was told to stop.
        """
        return self.state >= 2

    def batch_results(self, results):
        logging.debug("batch_results() invoked")
        queue = self.mgr.get_queue(self.qname_out)
        for item in results:
            queue.put(item, block=True)
        logging.debug("batch_results() returning data")

    def terminate(self):
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

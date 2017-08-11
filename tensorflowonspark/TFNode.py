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
    """
    Invoked from the user-supplied TensorFlow main function, which should have been launched as a background thread in the start() method
    with a multiprocessing.Manager as an argument.  This Manager and a unique queue name must be supplied to this function.
    DEPRECATED: Use TFNode class instead.
    """
    logging.debug("next_batch() invoked")
    queue = mgr.get_queue(qname)
    batch = []
    while len(batch) < batch_size:
        item = queue.get(block=True)
        if item is None:
            logging.info("next_batch() got None")
            queue.task_done()
            break
        elif type(item) is marker.EndPartition:
            logging.info("next_batch() got EndPartition")
            queue.task_done()
            if len(batch) > 0:
                break
        else:
            # logging.info("next_batch() got {0}".format(item))
            batch.append(item)
            queue.task_done()

    logging.debug("next_batch() returning data")
    return batch

def sig_def(signature_args):
    """Parses simplified-JSON signature_def from command-line args"""
    sdef = {}
    for s in signature_args:
        name = s.keys()[0]      # each sig has one top-level key (name)
        sig = s[name]     # where the value is the signature of inputs/outputs
        print(name)
        inputs = []
        for instr in sig['inputs']:                 # 'inputs' is an ordered list
            print("input: {}".format(instr))
            in_arr = instr.split('=')
            t = in_arr * 2 if len(in_arr) == 1 else in_arr
            if ':' not in t[1]:
              t[1] = t[1] + ':0'                    # add ':0' suffix to identify output tensor of op
            inputs.append(tuple(t))                 # split name:tensor
        sdef[name] = {}
        sdef[name]['inputs'] = inputs
        outputs = []
        for outstr in sig['outputs']:
            print("output: {}".format(outstr))
            out_arr = outstr.split('=')
            t = out_arr * 2 if len(out_arr) == 1 else out_arr
            if ':' not in t[1]:
              t[1] = t[1] + ':0'                    # add ':0' suffix to identify output tensor of op
            outputs.append(tuple(t))
        sdef[name]['outputs'] = outputs
    return sdef

def export_saved_model(sess, export_dir, tag_set, signatures):
    """Convenience function to export a saved_model using provided arguments"""
    import tensorflow as tf
    g = sess.graph
    g._unsafe_unfinalize()           # https://github.com/tensorflow/serving/issues/363
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    logging.info("===== signatures: {}".format(signatures))
    signature_def_map = {}
    for key, sig in signatures.items():
        signature_def_map[key] = tf.saved_model.signature_def_utils.build_signature_def(
                  inputs={ name:tf.saved_model.utils.build_tensor_info(g.get_tensor_by_name(tensorname)) for name, tensorname in sig['inputs'] },
                  outputs={ name:tf.saved_model.utils.build_tensor_info(g.get_tensor_by_name(tensorname)) for name, tensorname in sig['outputs'] },
                  method_name=sig['method_name'] if 'method_name' in sig else key)
    logging.info("===== signature_def_map: {}".format(signature_def_map))
    builder.add_meta_graph_and_variables(sess,
                  tag_set.split(','),
                  signature_def_map=signature_def_map,
                  clear_devices=True)
    g.finalize()
    builder.save()

def batch_results(mgr, results, qname='output'):
    """DEPRECATED: Use TFNode class instead"""
    logging.debug("batch_results() invoked")
    queue = mgr.get_queue(qname)
    for item in results:
        queue.put(item, block=True)
    logging.debug("batch_results() returning data")

def terminate(mgr, qname='input'):
    """DEPRECATED: Use TFNode class instead"""
    logging.info("terminate() invoked")
    mgr.set('state','terminating')

    # drop remaining items in the queue
    queue = mgr.get_queue(qname)
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

class DataFeed(object):
    def __init__(self, mgr, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):
        self.mgr = mgr
        self.train_mode = train_mode
        self.qname_in = qname_in
        self.qname_out = qname_out
        self.done_feeding = False
        self.input_mapping = [ x.split("=")[1] for x in input_mapping ] if input_mapping is not None else None

    def next_batch(self, batch_size):
        """
        Returns a batch of items from the input RDD as either an array or a dict (depending on the input_mapping).

        If multiple tensors are provided per row, e.g. tuple of (tensor1, tensor2, ..., tensorN) and no input_mapping
        is provided, the caller will be responsible for separating the tensors from the resulting array of tuples.

        If an input_mapping is provided to the DataFeed constructor, this will return a dictionary of tensors,
        where the tensors will be constructed/named in the same order as specified in the input_mapping.
        """
        logging.debug("next_batch() invoked")
        queue = self.mgr.get_queue(self.qname_in)
        tensors = [] if self.input_mapping is None else { t:[] for t in self.input_mapping }
        count = 0
        while count < batch_size:
            item = queue.get(block=True)
            if item is None:
                logging.info("next_batch() got None")
                queue.task_done()
                self.done_feeding = True
                break
            elif type(item) is marker.EndPartition:
                logging.info("next_batch() got EndPartition")
                queue.task_done()
                if not self.train_mode and count > 0:
                    break
            else:
                # logging.info("next_batch() got {0}".format(item))
                if self.input_mapping is None:
                  tensors.append(item)
                else:
                  for i in range(len(self.input_mapping)):
                    tensors[self.input_mapping[i]].append(item[i])
                count += 1
                queue.task_done()
        logging.debug("next_batch() returning {0} items".format(count))
        return tensors

    def should_stop(self):
        """Check if the feed process was told to stop."""
        return self.done_feeding

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


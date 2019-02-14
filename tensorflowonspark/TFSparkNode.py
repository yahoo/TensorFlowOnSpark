# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module provides low-level functions for managing the TensorFlowOnSpark cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import json
import logging
import multiprocessing
import os
import platform
import socket
import subprocess
import sys
import uuid
import time
import traceback
from threading import Thread

from . import TFManager
from . import TFNode
from . import gpu_info
from . import marker
from . import reservation
from . import util


class TFNodeContext:
  """Encapsulates unique metadata for a TensorFlowOnSpark node/executor and provides methods to interact with Spark and HDFS.

  An instance of this object will be passed to the TensorFlow "main" function via the `ctx` argument.
  To simply the end-user API, this class now mirrors the functions of the TFNode module.

  Args:
    :executor_id: integer identifier for this executor, per ``nodeRDD = sc.parallelize(range(num_executors), num_executors).``
    :job_name: TensorFlow job name (e.g. 'ps' or 'worker') of this TF node, per cluster_spec.
    :task_index: integer rank per job_name, e.g. "worker:0", "worker:1", "ps:0".
    :cluster_spec: dictionary for constructing a tf.train.ClusterSpec.
    :defaultFS: string representation of default FileSystem, e.g. ``file://`` or ``hdfs://<namenode>:8020/``.
    :working_dir: the current working directory for local filesystems, or YARN containers.
    :mgr: TFManager instance for this Python worker.
  """
  def __init__(self, executor_id, job_name, task_index, cluster_spec, defaultFS, working_dir, mgr):
    self.worker_num = executor_id       # for backwards-compatibility
    self.executor_id = executor_id
    self.job_name = job_name
    self.task_index = task_index
    self.cluster_spec = cluster_spec
    self.defaultFS = defaultFS
    self.working_dir = working_dir
    self.mgr = mgr

  def absolute_path(self, path):
    """Convenience function to access ``TFNode.hdfs_path`` directly from this object instance."""
    return TFNode.hdfs_path(self, path)

  def start_cluster_server(self, num_gpus=1, rdma=False):
    """Convenience function to access ``TFNode.start_cluster_server`` directly from this object instance."""
    return TFNode.start_cluster_server(self, num_gpus, rdma)

  def export_saved_model(self, sess, export_dir, tag_set, signatures):
    """Convenience function to access ``TFNode.export_saved_model`` directly from this object instance."""
    TFNode.export_saved_model(sess, export_dir, tag_set, signatures)

  def get_data_feed(self, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):
    """Convenience function to access ``TFNode.DataFeed`` directly from this object instance."""
    return TFNode.DataFeed(self.mgr, train_mode, qname_in, qname_out, input_mapping)


class TFSparkNode(object):
  """Low-level functions used by the high-level TFCluster APIs to manage cluster state.

  **This class is not intended for end-users (see TFNode for end-user APIs)**.

  For cluster management, this wraps the per-node cluster logic as Spark RDD mapPartitions functions, where the RDD is expected to be
  a "nodeRDD" of the form: ``nodeRDD = sc.parallelize(range(num_executors), num_executors)``.

  For data feeding, this wraps the feeding logic as Spark RDD mapPartitions functions on a standard "dataRDD".

  This also manages a reference to the TFManager "singleton" per executor.  Since Spark can spawn more than one python-worker
  per executor, this will reconnect to the "singleton" instance as needed.
  """
  mgr = None                #: TFManager instance
  cluster_id = None         #: Unique ID for a given TensorFlowOnSpark cluster, used for invalidating state for new clusters.


def _get_manager(cluster_info, host, executor_id):
  """Returns this executor's "singleton" instance of the multiprocessing.Manager, reconnecting per python-worker if needed.

  Args:
    :cluster_info: cluster node reservations
    :host: host IP address
    :executor_id: unique id per executor (created during initial call to run())

  Returns:
    TFManager instance for this executor/python-worker
  """
  for node in cluster_info:
    if node['host'] == host and node['executor_id'] == executor_id:
      addr = node['addr']
      authkey = node['authkey']
      TFSparkNode.mgr = TFManager.connect(addr, authkey)
      break

  if TFSparkNode.mgr is None:
    msg = "No TFManager found on this node, please ensure that:\n" + \
          "1. Spark num_executors matches TensorFlow cluster_size\n" + \
          "2. Spark cores/tasks per executor is 1.\n" + \
          "3. Spark dynamic allocation is disabled."
    raise Exception(msg)

  logging.info("Connected to TFSparkNode.mgr on {0}, executor={1}, state={2}".format(host, executor_id, str(TFSparkNode.mgr.get('state'))))
  return TFSparkNode.mgr


def run(fn, tf_args, cluster_meta, tensorboard, log_dir, queues, background):
  """Wraps the user-provided TensorFlow main function in a Spark mapPartitions function.

  Args:
    :fn: TensorFlow "main" function provided by the user.
    :tf_args: ``argparse`` args, or command line ``ARGV``.  These will be passed to the ``fn``.
    :cluster_meta: dictionary of cluster metadata (e.g. cluster_id, reservation.Server address, etc).
    :tensorboard: boolean indicating if the chief worker should spawn a Tensorboard server.
    :log_dir: directory to save tensorboard event logs.  If None, defaults to a fixed path on local filesystem.
    :queues: *INTERNAL_USE*
    :background: boolean indicating if the TensorFlow "main" function should be run in a background process.

  Returns:
    A nodeRDD.mapPartitions() function.
  """
  def _mapfn(iter):
    import tensorflow as tf

    # Note: consuming the input iterator helps Pyspark re-use this worker,
    for i in iter:
      executor_id = i

    # check that there are enough available GPUs (if using tensorflow-gpu) before committing reservation on this node
    if tf.test.is_built_with_cuda():
      num_gpus = tf_args.num_gpus if 'num_gpus' in tf_args else 1
      gpus_to_use = gpu_info.get_gpus(num_gpus)

    # assign TF job/task based on provided cluster_spec template (or use default/null values)
    job_name = 'default'
    task_index = -1
    cluster_id = cluster_meta['id']
    cluster_template = cluster_meta['cluster_template']
    for jobtype in cluster_template:
      nodes = cluster_template[jobtype]
      if executor_id in nodes:
        job_name = jobtype
        task_index = nodes.index(executor_id)
        break

    # get unique key (hostname, executor_id) for this executor
    host = util.get_ip_address()
    util.write_executor_id(executor_id)
    port = 0

    # check for existing TFManagers
    if TFSparkNode.mgr is not None and str(TFSparkNode.mgr.get('state')) != "'stopped'":
      if TFSparkNode.cluster_id == cluster_id:
        # raise an exception to force Spark to retry this "reservation" task on another executor
        raise Exception("TFManager already started on {0}, executor={1}, state={2}".format(host, executor_id, str(TFSparkNode.mgr.get("state"))))
      else:
        # old state, just continue with creating new manager
        logging.warn("Ignoring old TFManager with cluster_id {0}, requested cluster_id {1}".format(TFSparkNode.cluster_id, cluster_id))

    # start a TFManager and get a free port
    # use a random uuid as the authkey
    authkey = uuid.uuid4().bytes
    addr = None
    if job_name in ('ps', 'evaluator'):
      # PS nodes must be remotely accessible in order to shutdown from Spark driver.
      TFSparkNode.mgr = TFManager.start(authkey, ['control', 'error'], 'remote')
      addr = (host, TFSparkNode.mgr.address[1])
    else:
      # worker nodes only need to be locally accessible within the executor for data feeding
      TFSparkNode.mgr = TFManager.start(authkey, queues)
      addr = TFSparkNode.mgr.address

    # initialize mgr state
    TFSparkNode.mgr.set('state', 'running')
    TFSparkNode.cluster_id = cluster_id

    # expand Hadoop classpath wildcards for JNI (Spark 2.x)
    if 'HADOOP_PREFIX' in os.environ:
      classpath = os.environ['CLASSPATH']
      hadoop_path = os.path.join(os.environ['HADOOP_PREFIX'], 'bin', 'hadoop')
      hadoop_classpath = subprocess.check_output([hadoop_path, 'classpath', '--glob']).decode()
      logging.debug("CLASSPATH: {0}".format(hadoop_classpath))
      os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath

    # start TensorBoard if requested
    tb_pid = 0
    tb_port = 0
    if tensorboard and job_name == 'worker' and task_index == 0:
      tb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      tb_sock.bind(('', 0))
      tb_port = tb_sock.getsockname()[1]
      tb_sock.close()
      logdir = log_dir if log_dir else "tensorboard_%d" % executor_id

      # search for tensorboard in python/bin, PATH, and PYTHONPATH
      pypath = sys.executable
      pydir = os.path.dirname(pypath)
      sys_path = os.pathsep.join(sys.path)
      search_path = os.pathsep.join([pydir, sys_path, os.environ['PATH'], os.environ['PYTHONPATH']])

      tb_path = util.find_in_path(search_path, 'tensorboard')                             # executable in PATH
      if not tb_path:
        tb_path = util.find_in_path(search_path, 'tensorboard/main.py')                   # TF 1.3+
      if not tb_path:
        tb_path = util.find_in_path(search_path, 'tensorflow/tensorboard/__main__.py')    # TF 1.2-
      if not tb_path:
        raise Exception("Unable to find 'tensorboard' in: {}".format(search_path))

      # launch tensorboard
      tb_proc = subprocess.Popen([pypath, tb_path, "--logdir=%s" % logdir, "--port=%d" % tb_port], env=os.environ)
      tb_pid = tb_proc.pid

    # check server to see if this task is being retried (i.e. already reserved)
    client = reservation.Client(cluster_meta['server_addr'])
    cluster_info = client.get_reservations()
    tmp_sock = None
    node_meta = None
    for node in cluster_info:
      (nhost, nexec) = (node['host'], node['executor_id'])
      if nhost == host and nexec == executor_id:
        node_meta = node
        port = node['port']

    # if not already done, register everything we need to set up the cluster
    if node_meta is None:
      # first, find a free port for TF
      tmp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      tmp_sock.bind(('', port))
      port = tmp_sock.getsockname()[1]

      node_meta = {
          'executor_id': executor_id,
          'host': host,
          'job_name': job_name,
          'task_index': task_index,
          'port': port,
          'tb_pid': tb_pid,
          'tb_port': tb_port,
          'addr': addr,
          'authkey': authkey
      }
      # register node metadata with server
      logging.info("TFSparkNode.reserve: {0}".format(node_meta))
      client.register(node_meta)
      # wait for other nodes to finish reservations
      cluster_info = client.await_reservations()
      client.close()

    # construct a TensorFlow clusterspec from cluster_info
    sorted_cluster_info = sorted(cluster_info, key=lambda k: k['executor_id'])
    cluster_spec = {}
    last_executor_id = -1
    for node in sorted_cluster_info:
      if (node['executor_id'] == last_executor_id):
        raise Exception("Duplicate worker/task in cluster_info")
      last_executor_id = node['executor_id']
      logging.info("node: {0}".format(node))
      (njob, nhost, nport) = (node['job_name'], node['host'], node['port'])
      hosts = [] if njob not in cluster_spec else cluster_spec[njob]
      hosts.append("{0}:{1}".format(nhost, nport))
      cluster_spec[njob] = hosts

    # update TF_CONFIG if cluster spec has a 'master' node (i.e. tf.estimator)
    if 'master' in cluster_spec or 'chief' in cluster_spec:
      tf_config = json.dumps({
        'cluster': cluster_spec,
        'task': {'type': job_name, 'index': task_index},
        'environment': 'cloud'
      })
      logging.info("export TF_CONFIG: {}".format(tf_config))
      os.environ['TF_CONFIG'] = tf_config

    # reserve GPU(s) again, just before launching TF process (in case situation has changed)
    if tf.test.is_built_with_cuda():
      # compute my index relative to other nodes on the same host (for GPU allocation)
      my_addr = cluster_spec[job_name][task_index]
      my_host = my_addr.split(':')[0]
      flattened = [v for sublist in cluster_spec.values() for v in sublist]
      local_peers = [p for p in flattened if p.startswith(my_host)]
      my_index = local_peers.index(my_addr)

      num_gpus = tf_args.num_gpus if 'num_gpus' in tf_args else 1
      gpus_to_use = gpu_info.get_gpus(num_gpus, my_index)
      gpu_str = "GPUs" if num_gpus > 1 else "GPU"
      logging.debug("Requested {} {}, setting CUDA_VISIBLE_DEVICES={}".format(num_gpus, gpu_str, gpus_to_use))
      os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    # create a context object to hold metadata for TF
    ctx = TFNodeContext(executor_id, job_name, task_index, cluster_spec, cluster_meta['default_fs'], cluster_meta['working_dir'], TFSparkNode.mgr)

    # release port reserved for TF as late as possible
    if tmp_sock is not None:
      tmp_sock.close()

    # Background mode relies reuse of python worker in Spark.
    if background:
      # However, reuse of python worker can't work on Windows, we need to check if the current
      # script runs on Windows or not.
      if os.name == 'nt' or platform.system() == 'Windows':
        raise Exception("Background mode is not supported on Windows.")
      # Check if the config of reuse python worker is enabled on Spark.
      if not os.environ.get("SPARK_REUSE_WORKER"):
        raise Exception("Background mode relies reuse of python worker on Spark. This config 'spark.python.worker.reuse' is not enabled on Spark. Please enable it before using background.")

    def wrapper_fn(args, context):
      """Wrapper function that sets the sys.argv of the executor."""
      if isinstance(args, list):
        sys.argv = args
      fn(args, context)

    def wrapper_fn_background(args, context):
      """Wrapper function that signals exceptions to foreground process."""
      errq = TFSparkNode.mgr.get_queue('error')
      try:
        wrapper_fn(args, context)
      except Exception:
        errq.put(traceback.format_exc())
        errq.join()

    if job_name in ('ps', 'evaluator') or background:
      # invoke the TensorFlow main function in a background thread
      logging.info("Starting TensorFlow {0}:{1} as {2} on cluster node {3} on background process".format(
        job_name, task_index, job_name, executor_id))

      p = multiprocessing.Process(target=wrapper_fn_background, args=(tf_args, ctx))
      if job_name in ('ps','evaluator'):
        p.daemon = True
      p.start()

      # for ps nodes only, wait indefinitely in foreground thread for a "control" event (None == "stop")
      if job_name in ('ps', 'evaluator'):
        queue = TFSparkNode.mgr.get_queue('control')
        equeue = TFSparkNode.mgr.get_queue('error')
        done = False
        while not done:
          while (queue.empty() and equeue.empty()):
            time.sleep(1)
          if (not equeue.empty()):
            e_str = equeue.get()
            equeue.task_done()
            raise Exception("exception in " + job_name + ":\n" + e_str)
          msg = queue.get(block=True)
          logging.info("Got msg: {0}".format(msg))
          if msg is None:
            logging.info("Terminating {}".format(job_name))
            TFSparkNode.mgr.set('state', 'stopped')
            done = True
          queue.task_done()
    else:
      # otherwise, just run TF function in the main executor/worker thread
      logging.info("Starting TensorFlow {0}:{1} on cluster node {2} on foreground thread".format(job_name, task_index, executor_id))
      wrapper_fn(tf_args, ctx)
      logging.info("Finished TensorFlow {0}:{1} on cluster node {2}".format(job_name, task_index, executor_id))

  return _mapfn


def train(cluster_info, cluster_meta, feed_timeout=600, qname='input'):
  """Feeds Spark partitions into the shared multiprocessing.Queue.

  Args:
    :cluster_info: node reservation information for the cluster (e.g. host, executor_id, pid, ports, etc)
    :cluster_meta: dictionary of cluster metadata (e.g. cluster_id, reservation.Server address, etc)
    :feed_timeout: number of seconds after which data feeding times out (600 sec default)
    :qname: *INTERNAL_USE*

  Returns:
    A dataRDD.mapPartitions() function
  """
  def _train(iter):
    # get shared queue, reconnecting if necessary
    mgr = _get_manager(cluster_info, util.get_ip_address(), util.read_executor_id())
    try:
      queue = mgr.get_queue(qname)
      equeue = mgr.get_queue('error')
    except (AttributeError, KeyError):
      msg = "Queue '{}' not found on this node, check for exceptions on other nodes.".format(qname)
      raise Exception(msg)

    state = str(mgr.get('state'))
    logging.info("mgr.state={0}".format(state))
    terminating = state == "'terminating'"
    if terminating:
      logging.info("mgr is terminating, skipping partition")
      count = sum(1 for item in iter)
      logging.info("Skipped {0} items from partition".format(count))
    else:
      logging.info("Feeding partition {0} into {1} queue {2}".format(iter, qname, queue))
      count = 0
      for item in iter:
        count += 1
        queue.put(item, block=True)

      # wait for consumers to finish processing all items in queue before "finishing" this iterator
      joinThr = Thread(target=queue.join)
      joinThr.start()
      timeout = feed_timeout
      while (joinThr.isAlive()):
        if (not equeue.empty()):
          e_str = equeue.get()
          equeue.task_done()
          raise Exception("exception in worker:\n" + e_str)
        time.sleep(1)
        timeout -= 1
        if timeout <= 0:
          raise Exception("Timeout while feeding partition")

      logging.info("Processed {0} items in partition".format(count))

    # check if TF is terminating feed after this partition
    if not terminating:
      state = str(mgr.get('state'))
      terminating = state == "'terminating'"
      if terminating:
        try:
          logging.info("TFSparkNode: requesting stop")
          client = reservation.Client(cluster_meta['server_addr'])
          client.request_stop()
          client.close()
        except Exception as e:
          # ignore any errors while requesting stop
          logging.debug("Error while requesting stop: {0}".format(e))

    return [terminating]

  return _train


def inference(cluster_info, feed_timeout=600, qname='input'):
  """Feeds Spark partitions into the shared multiprocessing.Queue and returns inference results.

  Args:
    :cluster_info: node reservation information for the cluster (e.g. host, executor_id, pid, ports, etc)
    :feed_timeout: number of seconds after which data feeding times out (600 sec default)
    :qname: *INTERNAL_USE*

  Returns:
    A dataRDD.mapPartitions() function
  """
  def _inference(iter):
    # get shared queue, reconnecting if necessary
    mgr = _get_manager(cluster_info, util.get_ip_address(), util.read_executor_id())
    try:
      queue_in = mgr.get_queue(qname)
      equeue = mgr.get_queue('error')
    except (AttributeError, KeyError):
      msg = "Queue '{}' not found on this node, check for exceptions on other nodes.".format(qname)
      raise Exception(msg)

    logging.info("Feeding partition {0} into {1} queue {2}".format(iter, qname, queue_in))
    count = 0
    for item in iter:
      count += 1
      queue_in.put(item, block=True)

    # signal "end of partition"
    queue_in.put(marker.EndPartition())

    # skip empty partitions
    if count == 0:
      return []

    # wait for consumers to finish processing all items in queue before "finishing" this iterator
    joinThr = Thread(target=queue_in.join)
    joinThr.start()
    timeout = feed_timeout
    while (joinThr.isAlive()):
      if (not equeue.empty()):
        e_str = equeue.get()
        equeue.task_done()
        raise Exception("exception in worker:\n" + e_str)
      time.sleep(1)
      timeout -= 1
      if timeout <= 0:
        raise Exception("Timeout while feeding partition")

    logging.info("Processed {0} items in partition".format(count))

    # read result queue
    results = []
    queue_out = mgr.get_queue('output')
    while count > 0:
      result = queue_out.get(block=True)
      results.append(result)
      count -= 1
      queue_out.task_done()

    logging.info("Finished processing partition")
    return results

  return _inference


def shutdown(cluster_info, queues=['input']):
  """Stops all TensorFlow nodes by feeding ``None`` into the multiprocessing.Queues.

  Args:
    :cluster_info: node reservation information for the cluster (e.g. host, executor_id, pid, ports, etc).
    :queues: *INTERNAL_USE*

  Returns:
    A nodeRDD.mapPartitions() function
  """
  def _shutdown(iter):
    host = util.get_ip_address()
    executor_id = util.read_executor_id()

    # reconnect to shared queue
    mgr = _get_manager(cluster_info, host, executor_id)

    # send SIGTERM to Tensorboard proc (if running)
    for node in cluster_info:
      if node['host'] == host and node['executor_id'] == executor_id:
        tb_pid = node['tb_pid']
        if tb_pid != 0:
          logging.info("Stopping tensorboard (pid={0})".format(tb_pid))
          subprocess.Popen(["kill", str(tb_pid)])

    # terminate any listening queues
    logging.info("Stopping all queues")
    for q in queues:
      try:
        queue = mgr.get_queue(q)
        logging.info("Feeding None into {0} queue".format(q))
        queue.put(None, block=True)
      except (AttributeError, KeyError):
        msg = "Queue '{}' not found on this node, check for exceptions on other nodes.".format(q)
        raise Exception(msg)

    logging.info("Setting mgr.state to 'stopped'")
    mgr.set('state', 'stopped')
    return [True]

  return _shutdown

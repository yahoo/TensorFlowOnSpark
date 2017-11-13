# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module provides low-level functions for managing the TensorFlowOnSpark cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import os
import platform
import socket
import subprocess
import sys
import multiprocessing
import uuid
from . import TFManager
from . import reservation
from . import marker
from . import util

class TFNodeContext:
  """Struct to encapsulate unique metadata for a TensorFlowOnSpark node/executor.

  An instance of this object will be passed to the TensorFlow "main" function via the `ctx` argument.

  Args:
    :worker_num: integer identifier for this executor, per ``nodeRDD = sc.parallelize(range(num_executors), num_executors).``
    :job_name: TensorFlow job name (e.g. 'ps' or 'worker') of this TF node, per cluster_spec.
    :task_index: integer rank per job_name, e.g. "worker:0", "worker:1", "ps:0".
    :cluster_spec: tf.train.ClusterSpec
    :defaultFS: string representation of default FileSystem, e.g. ``file://`` or ``hdfs://<namenode>:8020/``.
    :working_dir: the current working directory for local filesystems, or YARN containers.
    :mgr: TFManager instance for this Python worker.
  """
  def __init__(self, worker_num, job_name, task_index, cluster_spec, defaultFS, working_dir, mgr):
    self.worker_num = worker_num
    self.job_name = job_name
    self.task_index = task_index
    self.cluster_spec = cluster_spec
    self.defaultFS = defaultFS
    self.working_dir = working_dir
    self.mgr = mgr

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

def _get_manager(cluster_info, host, ppid):
  """Returns this executor's "singleton" instance of the multiprocessing.Manager, reconnecting per python-worker if needed.

  Args:
    :cluster_info: cluster node reservations
    :host: host IP
    :ppid: parent (executor JVM) PID

  Returns:
    TFManager instance for this executor/python-worker
  """
  for node in cluster_info:
    if node['host'] == host and node['ppid'] == ppid:
      addr = node['addr']
      authkey = node['authkey']
      TFSparkNode.mgr = TFManager.connect(addr,authkey)
      break
  logging.info("Connected to TFSparkNode.mgr on {0}, ppid={1}, state={2}".format(host, ppid, str(TFSparkNode.mgr.get('state'))))
  return TFSparkNode.mgr

def reserve(cluster_spec, tensorboard, cluster_id, log_dir=None, queues=['input', 'output']):
  """*DEPRECATED*. use run() method instead of reserve/start."""
  raise Exception("DEPRECATED: use run() method instead of reserve/start")

def start(fn, tf_args, cluster_info, defaultFS, working_dir, background):
  """*DEPRECATED*. use run() method instead of reserve/start."""
  raise Exception("DEPRECATED: use run() method instead of reserve/start")

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
    # Note: consuming the input iterator helps Pyspark re-use this worker,
    for i in iter:
      worker_num = i

    # assign TF job/task based on provided cluster_spec template (or use default/null values)
    job_name = 'default'
    task_index = -1
    cluster_id = cluster_meta['id']
    cluster_template = cluster_meta['cluster_template']
    for jobtype in cluster_template:
      nodes = cluster_template[jobtype]
      if worker_num in nodes:
        job_name = jobtype
        task_index = nodes.index(worker_num)
        break

    # get unique id (hostname,ppid) for this executor's JVM
    host = util.get_ip_address()
    ppid = os.getppid()
    port = 0

    # check for existing TFManagers
    if TFSparkNode.mgr is not None and str(TFSparkNode.mgr.get('state')) != "'stopped'":
      if TFSparkNode.cluster_id == cluster_id:
        # raise an exception to force Spark to retry this "reservation" task on another executor
        raise Exception("TFManager already started on {0}, ppid={1}, state={2}".format(host, ppid, str(TFSparkNode.mgr.get("state"))))
      else:
        # old state, just continue with creating new manager
        logging.warn("Ignoring old TFManager with cluster_id {0}, requested cluster_id {1}".format(TFSparkNode.cluster_id, cluster_id))

    # start a TFManager and get a free port
    # use a random uuid as the authkey
    authkey = uuid.uuid4().bytes
    addr = None
    if job_name == 'ps':
      # PS nodes must be remotely accessible in order to shutdown from Spark driver.
      TFSparkNode.mgr = TFManager.start(authkey, ['control'], 'remote')
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
      tb_sock.bind(('',0))
      tb_port = tb_sock.getsockname()[1]
      tb_sock.close()
      logdir = log_dir if log_dir else "tensorboard_%d" % worker_num

      # search for tensorboard in python/bin, PATH, and PYTHONPATH
      pypath = sys.executable
      pydir = os.path.dirname(pypath)
      search_path = os.pathsep.join([pydir, os.environ['PATH'], os.environ['PYTHONPATH']])
      tb_path = util.find_in_path(search_path, 'tensorboard')
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
      (nhost, nppid) = (node['host'], node['ppid'])
      if nhost == host and nppid == ppid:
        node_meta = node
        port = node['port']

    # if not already done, register everything we need to set up the cluster
    if node_meta is None:
      # first, find a free port for TF
      tmp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      tmp_sock.bind(('',port))
      port = tmp_sock.getsockname()[1]

      node_meta = {
          'worker_num': worker_num,
          'host': host,
          'ppid': ppid,
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
    sorted_cluster_info = sorted(cluster_info, key=lambda k: k['worker_num'])
    spec = {}
    for node in sorted_cluster_info:
      logging.info("node: {0}".format(node))
      (njob, nhost, nport) = (node['job_name'], node['host'], node['port'])
      hosts = [] if njob not in spec else spec[njob]
      hosts.append("{0}:{1}".format(nhost, nport))
      spec[njob] = hosts

    # create a context object to hold metadata for TF
    ctx = TFNodeContext(worker_num, job_name, task_index, spec, cluster_meta['default_fs'], cluster_meta['working_dir'], TFSparkNode.mgr)

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

    if job_name == 'ps' or background:
      # invoke the TensorFlow main function in a background thread
      logging.info("Starting TensorFlow {0}:{1} on cluster node {2} on background process".format(job_name, task_index, worker_num))
      p = multiprocessing.Process(target=fn, args=(tf_args, ctx))
      p.start()

      # for ps nodes only, wait indefinitely in foreground thread for a "control" event (None == "stop")
      if job_name == 'ps':
        queue = TFSparkNode.mgr.get_queue('control')
        done = False
        while not done:
          msg = queue.get(block=True)
          logging.info("Got msg: {0}".format(msg))
          if msg is None:
            logging.info("Terminating PS")
            TFSparkNode.mgr.set('state', 'stopped')
            done = True
          queue.task_done()
    else:
      # otherwise, just run TF function in the main executor/worker thread
      logging.info("Starting TensorFlow {0}:{1} on cluster node {2} on foreground thread".format(job_name, task_index, worker_num))
      fn(tf_args, ctx)
      logging.info("Finished TensorFlow {0}:{1} on cluster node {2}".format(job_name, task_index, worker_num))

  return _mapfn

def train(cluster_info, cluster_meta, qname='input'):
  """Feeds Spark partitions into the shared multiprocessing.Queue.

  Args:
    :cluster_info: node reservation information for the cluster (e.g. host, ppid, pid, ports, etc)
    :cluster_meta: dictionary of cluster metadata (e.g. cluster_id, reservation.Server address, etc)
    :qname: *INTERNAL_USE*

  Returns:
    A dataRDD.mapPartitions() function
  """
  def _train(iter):
    # get shared queue, reconnecting if necessary
    mgr = _get_manager(cluster_info, util.get_ip_address(), os.getppid())
    queue = mgr.get_queue(qname)
    state = str(mgr.get('state'))
    logging.info("mgr.state={0}".format(state))
    terminating = state == "'terminating'"
    if terminating:
      logging.info("mgr is terminating, skipping partition")
      count = 0
      for item in iter:
        count += 1
      logging.info("Skipped {0} items from partition".format(count))

    else:
      logging.info("Feeding partition {0} into {1} queue {2}".format(iter, qname, queue))
      count = 0
      for item in iter:
        count += 1
        queue.put(item, block=True)
      # wait for consumers to finish processing all items in queue before "finishing" this iterator
      queue.join()
      logging.info("Processed {0} items in partition".format(count))

    # check if TF is terminating feed after this partition
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

def inference(cluster_info, qname='input'):
  """Feeds Spark partitions into the shared multiprocessing.Queue and returns inference results.

  Args:
    :cluster_info: node reservation information for the cluster (e.g. host, ppid, pid, ports, etc)
    :qname: *INTERNAL_USE*

  Returns:
    A dataRDD.mapPartitions() function
  """
  def _inference(iter):
    # get shared queue, reconnecting if necessary
    mgr = _get_manager(cluster_info, util.get_ip_address(), os.getppid())
    queue_in = mgr.get_queue(qname)

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
    queue_in.join()
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
    :cluster_info: node reservation information for the cluster (e.g. host, ppid, pid, ports, etc).
    :queues: *INTERNAL_USE*

  Returns:
    A nodeRDD.mapPartitions() function
  """
  def _shutdown(iter):
    host = util.get_ip_address()
    ppid = os.getppid()

    # reconnect to shared queue
    mgr = _get_manager(cluster_info, host, ppid)

    # send SIGTERM to Tensorboard proc (if running)
    for node in cluster_info:
      if node['host'] == host and node['ppid'] == ppid:
        tb_pid = node['tb_pid']
        if tb_pid != 0:
          logging.info("Stopping tensorboard (pid={0})".format(tb_pid))
          subprocess.Popen(["kill", str(tb_pid)])

    # terminate any listening queues
    logging.info("Stopping all queues")
    for q in queues:
      queue = mgr.get_queue(q)
      logging.info("Feeding None into {0} queue".format(q))
      queue.put(None, block=True)

    logging.info("Setting mgr.state to 'stopped'")
    mgr.set('state', 'stopped')
    return [True]

  return _shutdown


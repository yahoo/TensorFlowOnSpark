# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

"""
This module provides Spark-compatible functions to launch TensorFlow on the executors.

There are three main phases of operation:
1. Reservation - reserves a port for the TensorFlow process on each executor and also starts a multiprocessing.Manager to
listen for data/control messages.  For TensorFlow cluster applications, a cluster_spec "template" should be supplied.
2. Startup - launches the Tensorflow main function on the executors.  Note: for cluster applications, this MUST be invoked from 
a background thread on the Spark driver because the PS nodes will block until the job completes.
3. Data feeding - sends RDD data to the TensorFlow nodes via each executor's multiprocessing.Manager.  Note: because the PS
nodes block on startup, they will not receive any RDD partitions.
4. Shutdown - sends a shutdown control message to the multiprocessing.Managers of the PS nodes.
"""

import logging
import os
import platform
import random
import socket
import subprocess
import threading
import time
import uuid
import Queue
import TFManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s (%(threadName)s-%(process)d) %(message)s",)

class TFNodeContext:
  """This encapsulates key metadata for each TF node"""
  def __init__(self, worker_num, job_name, task_index, cluster_spec, defaultFS, working_dir, mgr):
    self.worker_num = worker_num
    self.job_name = job_name
    self.task_index = task_index
    self.cluster_spec = cluster_spec
    self.defaultFS = defaultFS
    self.working_dir = working_dir
    self.mgr = mgr

class TFSparkNode(object):
    """
    This class contains the TFManager "singleton" per executor/python-worker.  Note that Spark may spawn more than one python-worker
    per executor, so these module functions will reconnect to the "singleton", if needed.
    """
    mgr = None
    cluster_id = None

def _get_manager(cluster_info, host, ppid):
    """
    Returns this executor's "singleton" instance of the multiprocessing.Manager, reconnecting per python-worker if needed.
    """
    for node in cluster_info:
        if node['host'] == host and node['ppid'] == ppid:
            addr = node['addr']
            authkey = node['authkey']
            TFSparkNode.mgr = TFManager.connect(addr,authkey)
            break;
    logging.info("Connected to TFSparkNode.mgr on {0}, ppid={1}, state={2}".format(host, ppid, str(TFSparkNode.mgr.get('state'))))
    return TFSparkNode.mgr

def reserve(cluster_spec, tensorboard, cluster_id, queues=['input', 'output']):
    """
    Allocates a port for Tensorflow on this node, starts TensorBoard if requested, and starts a multiprocessing.Manager to listen for data/control msgs.
    """
    def _reserve(iter):
        # worker_num is assigned for the cluster (and may not correlate to Spark's executor id)
        for i in iter:
            worker_num = i

        # assign TF job/task based on provided cluster_spec template (or use default/null values)
        job_name = 'default'
        task_index = -1
        for jobtype in cluster_spec:
            nodes = cluster_spec[jobtype]
            if worker_num in nodes:
               job_name = jobtype
               task_index = nodes.index(worker_num)
               break;

        # get unique id (hostname,ppid) for this executor's JVM
        host = socket.gethostname()
        ppid = os.getppid()

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
        authkey = uuid.uuid4()
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
        TFSparkNode.mgr.set('ppid', ppid)
        TFSparkNode.cluster_id = cluster_id

        # start TensorBoard if requested
        tb_pid = 0
        tb_port = 0
        if tensorboard and job_name == 'worker' and task_index == 0:
            tb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tb_sock.bind(('',0))
            tb_port = tb_sock.getsockname()[1]
            tb_sock.close()
            logdir = "tensorboard_%d" %(worker_num)

            if 'PYSPARK_PYTHON' in os.environ:
              # user-specified Python (typically Python.zip)
              pypath = os.environ['PYSPARK_PYTHON']
              logging.info("PYSPARK_PYTHON: {0}".format(pypath))
              pydir = os.path.dirname(pypath)
              tb_proc = subprocess.Popen([pypath, "%s/tensorboard"%pydir, "--logdir=%s"%logdir, "--port=%d"%tb_port, "--debug"])
            else:
              # system-installed Python & tensorboard
              tb_proc = subprocess.Popen(["tensorboard", "--logdir=%s"%logdir, "--port=%d"%tb_port, "--debug"])
            tb_pid = tb_proc.pid

        # find a free port for TF
        # TODO: bind to port until TF server start
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('',0))
        port = s.getsockname()[1]

        # sleep a bit to force Spark to distribute the remaining reservation tasks to other/idle executors
        time.sleep(10)

        s.close()

        # return everything we need to reconnect later
        resp = {
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
        logging.info("TFSparkNode.reserve: {0}".format(resp))
        return [resp]
    return _reserve

def start(fn, tf_args, cluster_info, defaultFS, working_dir, background):
    """
    Wraps the TensorFlow main function in a Spark mapPartitions-compatible function.
    """
    def _mapfn(iter):
        # Note: consuming the input iterator helps Pyspark re-use this worker,
        # but we'll use the worker_num assigned during the reserve() step.
        for i in iter:
            worker_num = i

        # construct a TensorFlow clusterspec from supplied cluster_info AND get node info for this executor
        # Note: we could compute the clusterspec outside this function, but it's just a subset of cluster_info...
        spec = {}
        host = socket.gethostname()
        ppid = os.getppid()
        job_name = ''
        task_index = -1

        for node in cluster_info:
            logging.info("node: {0}".format(node))
            (njob, nhost, nport, nppid) = (node['job_name'], node['host'], node['port'], node['ppid'])
            hosts = [] if njob not in spec else spec[njob]
            hosts.append("{0}:{1}".format(nhost, nport))
            spec[njob] = hosts
            if nhost == host and nppid == ppid:
                (worker_num, job_name, task_index) = (node['worker_num'], node['job_name'], node['task_index'])

        # figure out which executor we're on, and get the reference to the multiprocessing.Manager
        mgr = _get_manager(cluster_info, host, ppid)

        ctx = TFNodeContext(worker_num, job_name, task_index, spec, defaultFS, working_dir, mgr)

        # expand Hadoop classpath wildcards for JNI (Spark 2.x)
        if 'HADOOP_PREFIX' in os.environ:
            classpath = os.environ['CLASSPATH']
            hadoop_classpath = subprocess.check_output(['hadoop', 'classpath', '--glob'])
            os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath

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
            logging.info("Starting TensorFlow {0}:{1} on cluster node {2} on background thread".format(job_name, task_index, worker_num))
            t = threading.Thread(target=fn, args=(tf_args, ctx))
            t.start()

            # for ps nodes only, wait indefinitely for a "control" event (None == "stop")
            if job_name == 'ps':
                queue = mgr.get_queue('control')
                done = False
                while not done:
                    msg =  queue.get(block=True)
                    logging.info("Got msg: {0}".format(msg))
                    if msg == None:
                        logging.info("Terminating PS")
                        mgr.set('state', 'stopped')
                        done = True
                    queue.task_done()
        else:
            # otherwise, just run TF function in the main executor/worker thread
            logging.info("Starting TensorFlow {0}:{1} on cluster node {2} on foreground thread".format(job_name, task_index, worker_num))

            # package up the context for the TF node
            fn(tf_args, ctx)
            logging.info("Finished TensorFlow {0}:{1} on cluster node {2}".format(job_name, task_index, worker_num))

        return [(worker_num, job_name, task_index)]

    return _mapfn

def train(cluster_info, qname='input'):
    """
    Feeds Spark partitions into the shared multiprocessing.Queue.
    """
    def _train(iter):
        # get shared queue, reconnecting if necessary
        mgr = _get_manager(cluster_info, socket.gethostname(), os.getppid())
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
        return [terminating]

    return _train

def inference(cluster_info, qname='input'):
    """
    Feeds Spark partitions into the shared multiprocessing.Queue and returns inference results.
    """
    def _inference(iter):
        # get shared queue, reconnecting if necessary
        mgr = _get_manager(cluster_info, socket.gethostname(), os.getppid())
        queue_in = mgr.get_queue(qname)

        logging.info("Feeding partition {0} into {1} queue {2}".format(iter, qname, queue_in))
        count = 0
        for item in iter:
            count += 1
            queue_in.put(item, block=True)

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
    def _shutdown(iter):
        """
        Stops all TensorFlow nodes by feeding None into the multiprocessing.Queues.
        """
        host = socket.gethostname()
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


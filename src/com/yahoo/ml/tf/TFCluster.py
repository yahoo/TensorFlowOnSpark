# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""
This module provides a high-level API to manage the TensorFlowOnSpark cluster.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import getpass
import logging
import os
import random
import threading
import time
from pyspark.streaming import DStream
from . import reservation
from . import TFManager
from . import TFSparkNode

class InputMode(object):
    TENSORFLOW=0
    SPARK=1

class TFCluster(object):

    sc = None
    defaultFS = None
    working_dir = None
    num_executors = None
    nodeRDD = None
    cluster_id = None
    cluster_info = None
    cluster_meta = None
    input_mode = None
    queues = None
    server = None

    def start(self, map_fun, tf_args):
        """
        Starts the TensorFlow main function on each node/executor (per cluster_spec) in a background thread on the driver.
        DEPRECATED: use run() method instead of reserve/start.
        """
        logging.info("Starting TensorFlow")
        def _start():
            self.nodeRDD.foreachPartition(TFSparkNode.start(map_fun,
                                                            tf_args,
                                                            self.cluster_info,
                                                            self.defaultFS,
                                                            self.working_dir,
                                                            background=(self.input_mode == InputMode.SPARK)))

        # start TF on a background thread (on Spark driver)
        t = threading.Thread(target=_start)
        t.start()

        # sleep a bit to avoid having the next Spark job scheduled before the TFSparkNode.start() tasks on the background thread.
        time.sleep(5)

    def train(self, dataRDD, num_epochs=0, qname='input'):
        """
        Feeds Spark data partitions into the TensorFlow worker nodes.
        """
        logging.info("Feeding training data")
        assert(self.input_mode == InputMode.SPARK)
        assert(qname in self.queues)
        assert(num_epochs >= 0)

        if isinstance(dataRDD, DStream):
            # Spark Streaming
            dataRDD.foreachRDD(lambda rdd: rdd.foreachPartition(TFSparkNode.train(self.cluster_info, self.cluster_meta, qname)))
        else:
            # Spark RDD
            # if num_epochs unspecified, pick an arbitrarily "large" number for now
            # TODO: calculate via dataRDD.count() / batch_size / max_steps
            if num_epochs == 0:
                num_epochs = 10
            rdds = []
            for i in range(num_epochs):
                rdds.append(dataRDD)
            unionRDD = self.sc.union(rdds)
            unionRDD.foreachPartition(TFSparkNode.train(self.cluster_info, self.cluster_meta, qname))

    def inference(self, dataRDD, qname='input'):
        """
        Feeds Spark data partitions into the TensorFlow worker nodes and returns an RDD of results.
        """
        logging.info("Feeding inference data")
        assert(self.input_mode == InputMode.SPARK)
        assert(qname in self.queues)
        return dataRDD.mapPartitions(TFSparkNode.inference(self.cluster_info, qname))

    def shutdown(self, ssc=None):
        """
        Stops TensorFlow nodes
        """
        logging.info("Stopping TensorFlow nodes")

        # identify ps/workers
        ps_list, worker_list = [], []
        for node in self.cluster_info:
            if node['job_name'] == 'ps':
                ps_list.append(node)
            else:
                worker_list.append(node)

        if ssc is not None:
            # Spark Streaming
            done = False
            while not done:
                done = ssc.awaitTerminationOrTimeout(1)
                if not done and self.server.done:
                    logging.info("Server done, stopping StreamingContext")
                    ssc.stop(stopSparkContext=False, stopGraceFully=True)
                done = done or self.server.done
        else:
            # in TENSORFLOW mode, there is no "data feeding" job, only a "start" job, so we must wait for the TensorFlow workers
            # to complete all tasks, while accounting for any PS tasks which run indefinitely.
            if self.input_mode == InputMode.TENSORFLOW:
                count = 0
                done = False
                while not done:
                    st = self.sc.statusTracker()
                    jobs = st.getActiveJobsIds()
                    if len(jobs) > 0:
                        stages = st.getActiveStageIds()
                        for i in stages:
                            si = st.getStageInfo(i)
                            if si.numActiveTasks == len(ps_list):
                                # if we only have PS tasks left, check that we see this condition a couple times
                                count += 1
                                done = (count >= 3)
                                time.sleep(5)
                    else:
                        done = True

            # shutdown queues and managers for "worker" executors.
            # note: in SPARK mode, this job will immediately queue up behind the "data feeding" job.
            # in TENSORFLOW mode, this will only run after all workers have finished.
            workers = len(worker_list)
            workerRDD = self.sc.parallelize(range(workers), workers)
            workerRDD.foreachPartition(TFSparkNode.shutdown(self.cluster_info, self.queues))

        logging.info("Shutting down cluster")
        # shutdown queues and manageres for "PS" executors.
        # note: we have to connect/shutdown from the spark driver, because these executors are "busy" and won't accept any other tasks.
        for node in ps_list:
            addr = node['addr']
            authkey = node['authkey']
            m = TFManager.connect(addr, authkey)
            q = m.get_queue('control')
            q.put(None)
            q.join()

        # wait for all jobs to finish
        done = False
        while not done:
          time.sleep(5)
          st = self.sc.statusTracker()
          jobs = st.getActiveJobsIds()
          if len(jobs) == 0:
              break;

    def tensorboard_url(self):
        """
        Utility function to get Tensorboard URL
        """
        tb_url = None
        for node in self.cluster_info:
          if node['tb_port'] != 0:
            tb_url = "http://{0}:{1}".format(node['host'], node['tb_port'])
        return tb_url

def reserve(sc, num_executors, num_ps, tensorboard=False, input_mode=InputMode.TENSORFLOW, queues=['input','output']):
    """
    Reserves ports, starts a multiprocessing.Manager per executor, and starts TensorBoard on worker/0 if requested.
    DEPRECATED: use run() method instead of reserve/start.
    """
    logging.info("Reserving TFSparkNodes {0}".format("w/ TensorBoard" if tensorboard else ""))
    assert num_ps < num_executors

    # build a cluster_spec template using worker_nums
    spec = {}
    for i in range(num_executors):
        if i < num_ps:
            nodes = [] if 'ps' not in spec else spec['ps']
            nodes.append(i)
            spec['ps'] = nodes
        else:
            nodes = [] if 'worker' not in spec else spec['worker']
            nodes.append(i)
            spec['worker'] = nodes

    # get default filesystem from spark
    defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
    # strip trailing "root" slash from "file:///" to be consistent w/ "hdfs://..."
    if defaultFS.startswith("file://") and len(defaultFS) > 7 and defaultFS.endswith("/"):
        defaultFS = defaultFS[:-1]

    # create TFCluster object
    cluster_id = random.getrandbits(64)
    cluster = TFCluster()
    cluster.sc = sc
    cluster.defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
    cluster.working_dir = os.getcwd()
    cluster.nodeRDD = sc.parallelize(range(num_executors), num_executors)
    cluster.cluster_id = cluster_id
    cluster.input_mode = input_mode
    cluster.queues = queues
    cluster.cluster_info = cluster.nodeRDD.mapPartitions(TFSparkNode.reserve(spec, tensorboard, cluster_id, queues)).collect()

    # print cluster_info and extract TensorBoard URL
    tb_url = None
    for node in cluster.cluster_info:
      print(node)
      if node['tb_port'] != 0:
        tb_url = "http://{0}:{1}".format(node['host'], node['tb_port'])

    if tb_url is not None:
      logging.info("TensorBoard running at: {0}".format(tb_url))

    # since our "primary key" for each executor's TFManager is (host, ppid), sanity check for duplicates
    # Note: this may occur if Spark retries failed Python tasks on the same executor.
    tb_nodes = set()
    for node in cluster.cluster_info:
      node_id = (node['host'],node['ppid'])
      if node_id in tb_nodes:
        raise Exception("Duplicate cluster node id detected (host={0}, ppid={1}).  Please ensure that the number of executors >= number of TensorFlow nodes, and TFCluster.shutdown() is successfully invoked when done.".format(node_id[0], node_id[1]))
      else:
        tb_nodes.add(node_id)

    return cluster

def run(sc, map_fun, tf_args, num_executors, num_ps, tensorboard=False, input_mode=InputMode.TENSORFLOW, queues=['input', 'output']):
    """Runs TensorFlow processes on executors"""
    logging.info("Reserving TFSparkNodes {0}".format("w/ TensorBoard" if tensorboard else ""))
    assert num_ps < num_executors

    # build a cluster_spec template using worker_nums
    cluster_template = {}
    cluster_template['ps'] = range(num_ps)
    cluster_template['worker'] = range(num_ps, num_executors)

    # get default filesystem from spark
    defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
    # strip trailing "root" slash from "file:///" to be consistent w/ "hdfs://..."
    if defaultFS.startswith("file://") and len(defaultFS) > 7 and defaultFS.endswith("/"):
        defaultFS = defaultFS[:-1]

    # get current working dir of spark launch
    working_dir = os.getcwd()

    # start a server to listen for reservations and broadcast cluster_spec
    server = reservation.Server(num_executors)
    server_addr = server.start()

    # start TF nodes on all executors
    logging.info("Starting TensorFlow on executors")
    cluster_meta = {
      'id': random.getrandbits(64),
      'cluster_template': cluster_template,
      'num_executors': num_executors,
      'default_fs': defaultFS,
      'working_dir': working_dir,
      'server_addr': server_addr
    }
    nodeRDD = sc.parallelize(range(num_executors), num_executors)

    # start TF on a background thread (on Spark driver) to allow for feeding job
    def _start():
      nodeRDD.foreachPartition(TFSparkNode.run(map_fun,
                                                tf_args,
                                                cluster_meta,
                                                tensorboard,
                                                queues,
                                                background=(input_mode == InputMode.SPARK)))
    t = threading.Thread(target=_start)
    t.start()

    # wait for executors to register and start TFNodes before continuing
    logging.info("Waiting for TFSparkNodes to start")
    cluster_info = server.await_reservations()
    logging.info("All TFSparkNodes started")

    # print cluster_info and extract TensorBoard URL
    tb_url = None
    for node in cluster_info:
      logging.info(node)
      if node['tb_port'] != 0:
        tb_url = "http://{0}:{1}".format(node['host'], node['tb_port'])

    if tb_url is not None:
      logging.info("========================================================================================")
      logging.info("")
      logging.info("TensorBoard running at:       {0}".format(tb_url))
      logging.info("")
      logging.info("========================================================================================")

    # since our "primary key" for each executor's TFManager is (host, ppid), sanity check for duplicates
    # Note: this may occur if Spark retries failed Python tasks on the same executor.
    tb_nodes = set()
    for node in cluster_info:
      node_id = (node['host'],node['ppid'])
      if node_id in tb_nodes:
        raise Exception("Duplicate cluster node id detected (host={0}, ppid={1}).  Please ensure that (1) the number of executors >= number of TensorFlow nodes, (2) the number of tasks per executors == 1, and (3) TFCluster.shutdown() is successfully invoked when done.".format(node_id[0], node_id[1]))
      else:
        tb_nodes.add(node_id)

    # create TFCluster object
    cluster = TFCluster()
    cluster.sc = sc
    cluster.meta = cluster_meta
    cluster.nodeRDD = nodeRDD
    cluster.cluster_info = cluster_info
    cluster.cluster_meta = cluster_meta
    cluster.input_mode = input_mode
    cluster.queues = queues
    cluster.server = server

    return cluster

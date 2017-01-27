# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime

import os
import sys
import tensorflow as tf
import time

def main_fun(argv, ctx):

  # extract node metadata from ctx
  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  assert job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  from inception import inception_distributed_train
  from inception.imagenet_data import ImagenetData
  import tensorflow as tf

  # instantiate FLAGS on workers using argv from driver and add job_name and task_id
  print("argv:", argv)
  sys.argv = argv

  FLAGS = tf.app.flags.FLAGS
  FLAGS.job_name = job_name
  FLAGS.task_id = task_index
  print("FLAGS:", FLAGS.__dict__['__flags'])

  # Get TF cluster and server instances
  cluster_spec, server = TFNode.start_cluster_server(ctx, FLAGS.num_gpus, FLAGS.rdma)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec, ctx)

if __name__ == '__main__':
  # parse arguments needed by the Spark driver
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", help="number of epochs", type=int, default=0)
  parser.add_argument("--input_data", help="HDFS path to input dataset")
  parser.add_argument("--input_mode", help="method to ingest data: (spark|tf)", choices=["spark","tf"], default="tf")
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  (args,rem) = parser.parse_known_args()

  input_mode = TFCluster.InputMode.SPARK if args.input_mode == 'spark' else TFCluster.InputMode.TENSORFLOW

  print("{0} ===== Start".format(datetime.now().isoformat()))
  sc = SparkContext(conf=SparkConf().setAppName('imagenet_distributed_train'))
  num_executors = int(sc._conf.get("spark.executor.instances"))
  num_ps = 1

  cluster = TFCluster.reserve(sc, num_executors, num_ps, args.tensorboard, input_mode)
  cluster.start(main_fun, sys.argv)
  if input_mode == TFCluster.InputMode.SPARK:
    dataRDD = sc.newAPIHadoopFile(args.input_data, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")
    cluster.train(dataRDD, args.epochs)
  cluster.shutdown()
  print("{0} ===== Stop".format(datetime.now().isoformat()))

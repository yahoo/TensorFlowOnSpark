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
"""A binary to evaluate Inception on the flowers data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = 0.7874 recall @ 5 = 0.9436 [50000 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
import sys

def main_fun(argv, ctx):
  import tensorflow as tf
  from inception import inception_eval
  from inception.imagenet_data import ImagenetData

  print("argv:", argv)
  sys.argv = argv

  FLAGS = tf.app.flags.FLAGS
  FLAGS._parse_flags()
  print("FLAGS:", FLAGS.__dict__['__flags'])

  dataset = ImagenetData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)

  cluster_spec, server = TFNode.start_cluster_server(ctx, 1, FLAGS.rdma)

  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  sc = SparkContext(conf=SparkConf().setAppName("grid_imagenet_eval"))
  num_executors = int(sc._conf.get("spark.executor.instances"))
  num_ps = 0

  cluster = TFCluster.reserve(sc, num_executors, num_ps, False, TFCluster.InputMode.TENSORFLOW)
  cluster.start(main_fun, sys.argv)
  cluster.shutdown()

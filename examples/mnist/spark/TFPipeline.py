# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.ml.param.shared import *
from pyspark.ml.pipeline import Estimator, Model, Pipeline
from pyspark.sql import SparkSession

from tensorflowonspark import TFCluster

class TFEstimator(Estimator):
  train_fn = None
  infer_fn = None
  args = None
  input_mode = None

  def __init__(self, train_fn, infer_fn, args, input_mode):
    self.train_fn = train_fn
    self.infer_fn = infer_fn
    self.args = args
    self.input_mode = input_mode

  def _fit(self, dataset):
    self.args.mode = 'train'
    print("===== train args: {0}".format(self.args))
    sc = SparkContext.getOrCreate()
    cluster = TFCluster.run(sc, self.train_fn, self.args, self.args.cluster_size, self.args.num_ps, self.args.tensorboard, self.input_mode)
    cluster.train(dataset.rdd, self.args.epochs)
    cluster.shutdown()
    return TFModel(self.infer_fn, self.args)

class TFModel(Model):
  infer_fn = None
  args = None

  def __init__(self, infer_fn, args):
    self.infer_fn = infer_fn
    self.args = args

  def _transform(self, dataset):
    self.args.mode = 'inference'
    print("===== inference args: {0}".format(self.args))
    spark = SparkSession.builder.getOrCreate()
    rdd_out = dataset.rdd.mapPartitions(lambda it: self.infer_fn(self.args, it))
    return spark.createDataFrame(rdd_out, "string")

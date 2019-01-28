# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train DNN on census income dataset."""

import os

# from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers
import census_dataset
import wide_deep_run_loop


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='/tmp/census_data',
                          model_dir='/tmp/census_model',
                          train_epochs=40,
                          epochs_between_evals=2,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op, ctx):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  # Note: adding device_filter to fix: https://github.com/tensorflow/tensorflow/issues/21745
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0},
                                    device_filters=['/job:ps', '/job:%s/task:%d' % (ctx.job_name, ctx.task_index)],
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_census(flags_obj, ctx):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return census_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  # Removing run_loop, since we can only invoke train_and_evaluate once
  model_helpers.apply_clean(flags.FLAGS)
  model = build_estimator(
      model_dir=flags_obj.model_dir, model_type=flags_obj.model_type,
      model_column_fn=census_dataset.build_model_columns,
      inter_op=flags_obj.inter_op_parallelism_threads,
      intra_op=flags_obj.intra_op_parallelism_threads,
      ctx=ctx)

  loss_prefix = LOSS_PREFIX.get(flags_obj.model_type, '')
  tensors_to_log = {k: v.format(loss_prefix=loss_prefix)
                    for k, v in tensors_to_log.items()}
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks, model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size, tensors_to_log=tensors_to_log)

  # Note: this will only be invoked once, so `--epochs_between_evals` is now effectively `--train_epochs`
  # and evaluation will only execute once.
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main_fun(argv, ctx):
  sys.argv = argv
  define_census_flags()
  flags.FLAGS(sys.argv)
  tf.logging.set_verbosity(tf.logging.INFO)

  with logger.benchmark_context(flags.FLAGS):
    run_census(flags.FLAGS, ctx)


if __name__ == '__main__':
  import argparse
  import sys
  from pyspark import SparkConf, SparkContext
  from tensorflowonspark import TFCluster

  sc = SparkContext(conf=SparkConf().setAppName('wide_deep'))
  executors = int(sc._conf.get("spark.executor.instances", "1"))

  # arguments for Spark and TFoS
  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=executors)
  parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
  (args, remainder) = parser.parse_known_args()

  # construct an ARGV (with script name as first element) from remaining args and pass it to the TF processes on executors
  remainder.insert(0, __file__)
  print("spark args:", args)
  print("tf args:", remainder)

  num_workers = args.cluster_size - args.num_ps
  print("===== num_executors={}, num_workers={}, num_ps={}".format(args.cluster_size, num_workers, args.num_ps))

  cluster = TFCluster.run(sc, main_fun, remainder, args.cluster_size, args.num_ps, False, TFCluster.InputMode.TENSORFLOW, master_node='master')
  cluster.shutdown()

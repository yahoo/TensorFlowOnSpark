import argparse

import os
import random
import unittest
from tensorflowonspark import gpu_info, reservation, TFSparkNode
from unittest.mock import patch


class TFSparkNodeTest(unittest.TestCase):
  def setUp(self):
    self.server = reservation.Server(1)
    self.server_addr = self.server.start()

    self.parser = argparse.ArgumentParser()

    self.default_fn = lambda args, ctx: print("{}:{} args: {}".format(ctx.job_name, ctx.task_index, args))
    self.job_name = 'chief'
    self.task_index = 0
    self.cluster_meta = {
      'id': random.getrandbits(64),
      'cluster_template': {self.job_name: [0]},
      'num_executors': 1,
      'default_fs': 'file://',
      'working_dir': '.',
      'server_addr': self.server_addr
    }
    self.tensorboard = False
    self.log_dir = None
    self.queues = ['input']
    self.background = False

  def tearDown(self):
    client = reservation.Client(self.server_addr)
    client.request_stop()
    client.close()

  def test_run(self):
    """Minimal function w/ args and ctx"""
    def fn(args, ctx):
      print("{}:{} args: {}".format(ctx.job_name, ctx.task_index, args))
      self.assertEqual(ctx.job_name, self.job_name)
      self.assertEqual(ctx.task_index, 0)

    tf_args = self.parser.parse_args([])
    map_fn = TFSparkNode.run(fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])

  def test_gpu_unavailable(self):
    """Request GPU with no GPUs available, expecting an exception"""
    self.parser.add_argument("--num_gpus", help="number of gpus to use", type=int)
    tf_args = self.parser.parse_args(["--num_gpus", "1"])

    with self.assertRaises(Exception):
      map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
      map_fn([0])

  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_available(self, mock_available, mock_get_gpus):
    """Request available GPU"""
    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']

    self.parser.add_argument("--num_gpus", help="number of gpus to use", type=int)
    tf_args = self.parser.parse_args(["--num_gpus", "1"])

    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])
    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')

  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_default(self, mock_available, mock_get_gpus):
    """Default to one GPU if not explicitly requested"""

    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']

    tf_args = self.parser.parse_args([])

    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])
    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
    mock_get_gpus.assert_called_with(1, 0, format=gpu_info.AS_LIST)

  @patch('tensorflowonspark.TFSparkNode._get_cluster_spec')
  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_cluster_spec(self, mock_available, mock_get_gpus, mock_get_spec):
    """Request GPU when multiple TF nodes land on same executor"""

    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']
    mock_get_spec.return_value = {'chief': ['1.1.1.1:2222'], 'worker': ['1.1.1.1:2223', '1.1.1.1:2224', '2.2.2.2:2222']}

    self.cluster_meta['cluster_template'] = {'chief': [0], 'worker': [1, 2, 3]}
    self.parser.add_argument("--num_gpus", help="number of gpus to use", type=int)
    tf_args = self.parser.parse_args(["--num_gpus", "1"])
    print("tf_args:", tf_args)

    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([2])     # worker:1
    mock_get_gpus.assert_called_with(1, 2, format=gpu_info.AS_LIST)

  @patch('pyspark.TaskContext')
  @patch('tensorflowonspark.TFSparkNode._has_spark_resource_api')
  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_spark_available(self, mock_available, mock_get_gpus, mock_spark_resources, mock_context):
    """Spark resource API w/ available GPU"""
    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']
    mock_spark_resources.return_value = True
    mock_context_instance = mock_context.return_value
    mock_context_instance.resources.return_value = {'gpu': type("ResourceInformation", (object,), {"addresses": ['0']})}

    tf_args = self.parser.parse_args([])
    print("tf_args:", tf_args)
    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])

    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
    mock_get_gpus.assert_not_called()

  @patch('pyspark.TaskContext')
  @patch('tensorflowonspark.TFSparkNode._has_spark_resource_api')
  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_spark_fallback(self, mock_available, mock_get_gpus, mock_spark_resources, mock_context):
    """Spark resource API w/ no available GPU with fallback to original resource allocation"""
    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']
    mock_spark_resources.return_value = True
    mock_context_instance = mock_context.return_value
    mock_context_instance.resources.return_value = {}

    tf_args = self.parser.parse_args([])
    print("tf_args:", tf_args)
    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])

    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
    mock_get_gpus.assert_called_with(1, 0, format=gpu_info.AS_LIST)

  @patch.dict(os.environ, {'SPARK_EXECUTOR_POD_IP': '1.2.3.4'})
  @patch('pyspark.TaskContext')
  @patch('tensorflowonspark.TFSparkNode._has_spark_resource_api')
  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_spark_unavailable_default(self, mock_available, mock_get_gpus, mock_spark_resources, mock_context):
    """Spark resource API w/ no available GPU and no fallback (in K8s)"""
    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']
    mock_spark_resources.return_value = True
    mock_context_instance = mock_context.return_value
    mock_context_instance.resources.return_value = {}

    tf_args = self.parser.parse_args([])
    print("tf_args:", tf_args)

    map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
    map_fn([0])

    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
    mock_get_gpus.assert_not_called()

  @patch.dict(os.environ, {'SPARK_EXECUTOR_POD_IP': '1.2.3.4'})
  @patch('pyspark.TaskContext')
  @patch('tensorflowonspark.TFSparkNode._has_spark_resource_api')
  @patch('tensorflowonspark.gpu_info.get_gpus')
  @patch('tensorflowonspark.gpu_info.is_gpu_available')
  def test_gpu_spark_unavailable_but_requested(self, mock_available, mock_get_gpus, mock_spark_resources, mock_context):
    """Spark resource API w/ no available GPU and no fallback (in K8s) with num_gpus set"""
    mock_available.return_value = True
    mock_get_gpus.return_value = ['0']
    mock_spark_resources.return_value = True
    mock_context_instance = mock_context.return_value
    mock_context_instance.resources.return_value = {}

    self.parser.add_argument("--num_gpus", help="number of gpus to use", type=int)
    tf_args = self.parser.parse_args(["--num_gpus", "1"])
    print("tf_args:", tf_args)

    with self.assertRaises(Exception):
      map_fn = TFSparkNode.run(self.default_fn, tf_args, self.cluster_meta, self.tensorboard, self.log_dir, self.queues, self.background)
      map_fn([0])


if __name__ == '__main__':
  unittest.main()

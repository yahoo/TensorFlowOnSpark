import unittest
import test
from tensorflowonspark import TFParallel


class TFParallelTest(test.SparkTest):

  @classmethod
  def setUpClass(cls):
    super(TFParallelTest, cls).setUpClass()

  @classmethod
  def tearDownClass(cls):
    super(TFParallelTest, cls).tearDownClass()

  def test_basic_tf(self):
    """Single-node TF graph (w/ args) running independently on multiple executors."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      x = tf.constant(args['x'])
      y = tf.constant(args['y'])
      sum = tf.math.add(x, y)
      assert sum.numpy() == 3

    args = {'x': 1, 'y': 2}
    TFParallel.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers, use_barrier=False)

  def test_basic_tf_barrier(self):
    """Single-node TF graph (w/ args) running independently on multiple executors using Spark barrier."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      x = tf.constant(args['x'])
      y = tf.constant(args['y'])
      sum = tf.math.add(x, y)
      assert sum.numpy() == 3

    args = {'x': 1, 'y': 2}
    TFParallel.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers)

  def test_basic_tf_barrier_insufficient_resources(self):
    """Single-node TF graph (w/ args) running independently on multiple executors using Spark barrier with insufficient resource."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      x = tf.constant(args['x'])
      y = tf.constant(args['y'])
      sum = tf.math.add(x, y)
      assert sum.numpy() == 3

    args = {'x': 1, 'y': 2}
    with self.assertRaises(Exception):
      TFParallel.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers + 1)


if __name__ == '__main__':
  unittest.main()

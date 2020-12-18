import unittest
import test
import time
from tensorflowonspark import TFCluster, TFNode


class TFClusterTest(test.SparkTest):
  @classmethod
  def setUpClass(cls):
    super(TFClusterTest, cls).setUpClass()

  @classmethod
  def tearDownClass(cls):
    super(TFClusterTest, cls).tearDownClass()

  def test_basic_tf(self):
    """Single-node TF graph (w/ args) running independently on multiple executors."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      x = tf.constant(args['x'])
      y = tf.constant(args['y'])
      sum = tf.math.add(x, y)
      assert sum.numpy() == 3

    args = {'x': 1, 'y': 2}
    cluster = TFCluster.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers, num_ps=0)
    cluster.shutdown()

  def test_inputmode_spark(self):
    """Distributed TF cluster w/ InputMode.SPARK"""
    def _map_fun(args, ctx):
      import tensorflow as tf

      tf_feed = TFNode.DataFeed(ctx.mgr, False)
      while not tf_feed.should_stop():
        batch = tf_feed.next_batch(batch_size=10)
        print("batch: {}".format(batch))
        squares = tf.math.square(batch)
        print("squares: {}".format(squares))
        tf_feed.batch_results(squares.numpy())

    input = [[x] for x in range(1000)]    # set up input as tensors of shape [1] to match placeholder
    rdd = self.sc.parallelize(input, 10)
    cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.SPARK)
    rdd_out = cluster.inference(rdd)
    rdd_sum = rdd_out.sum()
    self.assertEqual(rdd_sum, sum([x * x for x in range(1000)]))
    cluster.shutdown()

  def test_inputmode_spark_exception(self):
    """Distributed TF cluster w/ InputMode.SPARK and exception during feeding"""
    def _map_fun(args, ctx):
      import tensorflow as tf

      tf_feed = TFNode.DataFeed(ctx.mgr, False)
      while not tf_feed.should_stop():
        batch = tf_feed.next_batch(10)
        if len(batch) > 0:
          squares = tf.math.square(batch)
          tf_feed.batch_results(squares.numpy())
          raise Exception("FAKE exception during feeding")

    input = [[x] for x in range(1000)]    # set up input as tensors of shape [1] to match placeholder
    rdd = self.sc.parallelize(input, 10)
    with self.assertRaises(Exception):
      cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.SPARK)
      cluster.inference(rdd, feed_timeout=1).count()
      cluster.shutdown()

  def test_inputmode_spark_late_exception(self):
    """Distributed TF cluster w/ InputMode.SPARK and exception after feeding"""
    def _map_fun(args, ctx):
      import tensorflow as tf

      tf_feed = TFNode.DataFeed(ctx.mgr, False)
      while not tf_feed.should_stop():
        batch = tf_feed.next_batch(10)
        if len(batch) > 0:
          squares = tf.math.square(batch)
          tf_feed.batch_results(squares.numpy())

      # simulate post-feed actions that raise an exception
      time.sleep(2)
      raise Exception("FAKE exception after feeding")

    input = [[x] for x in range(1000)]    # set up input as tensors of shape [1] to match placeholder
    rdd = self.sc.parallelize(input, 10)
    with self.assertRaises(Exception):
      cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.SPARK)
      cluster.inference(rdd).count()
      cluster.shutdown(grace_secs=5)      # note: grace_secs must be larger than the time needed for post-feed actions


if __name__ == '__main__':
  unittest.main()

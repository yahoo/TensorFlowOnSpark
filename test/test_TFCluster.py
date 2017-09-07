import unittest
import test
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
      sum = tf.add(x,y)
      with tf.Session() as sess:
        result = sess.run([sum])
        assert result[0] == 3

    args = { 'x':1, 'y':2 }
    cluster = TFCluster.run(self.sc, _map_fun, tf_args=args, num_executors=self.num_workers, num_ps=0)
    cluster.shutdown()

  def test_inputmode_spark(self):
    """Distributed TF cluster w/ InputMode.SPARK"""
    def _map_fun(args, ctx):
      import tensorflow as tf
      cluster, server = TFNode.start_cluster_server(ctx)
      if ctx.job_name == "ps":
        server.join()
      elif ctx.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % ctx.task_index,
          cluster=cluster)):
          x = tf.placeholder(tf.int32, [None, 1])
          sq = tf.square(x)
          init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=(ctx.task_index == 0),
                                init_op=init_op)
        with sv.managed_session(server.target) as sess:
          tf_feed = TFNode.DataFeed(ctx.mgr, False)
          while not sv.should_stop() and not tf_feed.should_stop():
            outputs = sess.run([sq], feed_dict={ x: tf_feed.next_batch(10) })
            tf_feed.batch_results(outputs[0])
        sv.stop()

    input = [ [x] for x in range(1000) ]    # set up input as tensors of shape [1] to match placeholder
    rdd = self.sc.parallelize(input, 10)
    cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.SPARK)
    rdd_out = cluster.inference(rdd)
    rdd_sum = rdd_out.sum()
    self.assertEqual(rdd_sum, sum( [x * x for x in range(1000)] ))
    cluster.shutdown()


if __name__ == '__main__':
  unittest.main()

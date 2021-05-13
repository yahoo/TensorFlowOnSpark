import os
import unittest
import test
import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from tensorflowonspark import TFCluster, TFNode


class TFClusterNoReuseWorker(test.SparkTest):
  """Tests which require spark.python.worker.reuse=False to avoid carrying TF distribution strategy state between tests."""
  @classmethod
  def setUpClass(cls):
    super(TFClusterNoReuseWorker, cls).setUpClass()

  @classmethod
  def getSparkConf(cls):
    return super(TFClusterNoReuseWorker, cls).getSparkConf().set('spark.python.worker.reuse', False)

  @classmethod
  def tearDownClass(cls):
    super(TFClusterNoReuseWorker, cls).tearDownClass()

  def test_mnist_tf(self):
    """Distributed TF Cluster / InputMode.TENSORFLOW for MNIST based on: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras"""
    def _map_fun(args, ctx):
      import tensorflow as tf
      import numpy as np

      #tf.keras.backend.clear_session()
      #tf.compat.v1.reset_default_graph()

      strategy = tf.distribute.MultiWorkerMirroredStrategy()

      def mnist_dataset(batch_size):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train / np.float32(255)
        y_train = y_train.astype(np.int64)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
        return train_dataset

      def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'])
        return model

      per_worker_batch_size = 64
      global_batch_size = per_worker_batch_size * ctx.num_workers
      multi_worker_dataset = mnist_dataset(global_batch_size)

      with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

      multi_worker_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=70)

    cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
    cluster.shutdown()

  def test_mnist_tf_release_port(self):
    """Distributed TF Cluster / InputMode.TENSORFLOW with reserved port/socket released by user code."""
    def _map_fun(args, ctx):
      import tensorflow as tf
      import numpy as np

      #tf.keras.backend.clear_session()
      #tf.compat.v1.reset_default_graph()

      # emulate "long" user process
      time.sleep(5)

      # release port before starting TF GRPC Server
      ctx.release_port()

      strategy = tf.distribute.MultiWorkerMirroredStrategy()

      def mnist_dataset(batch_size):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train / np.float32(255)
        y_train = y_train.astype(np.int64)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
        return train_dataset

      def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'])
        return model

      per_worker_batch_size = 64
      global_batch_size = per_worker_batch_size * ctx.num_workers
      multi_worker_dataset = mnist_dataset(global_batch_size)

      with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

      multi_worker_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=70)

    cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief', release_port=False)
    cluster.shutdown()

#  # This works in isolation, but impacts other tests (presumably due to the SystemExit)
#  def test_mnist_tf_unreleased_port(self):
#    """Distributed TF Cluster / InputMode.TENSORFLOW for MNIST w/ reserved port/socket not released by user code."""
#    def _map_fun(args, ctx):
#      import tensorflow as tf
#      import numpy as np
#
#      # emulate "long" user process
#      time.sleep(5)
#
#      # must release port before starting TF GRPC Server
#      # ctx.release_port()
#
#      # set comms timeout to 5 seconds (default: inf)
#      options = tf.distribute.experimental.CommunicationOptions(timeout_seconds=5)
#      strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options)
#
#      def mnist_dataset(batch_size):
#        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
#        x_train = x_train / np.float32(255)
#        y_train = y_train.astype(np.int64)
#        train_dataset = tf.data.Dataset.from_tensor_slices(
#            (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
#        return train_dataset
#
#      def build_and_compile_cnn_model():
#        model = tf.keras.Sequential([
#            tf.keras.Input(shape=(28, 28)),
#            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
#            tf.keras.layers.Conv2D(32, 3, activation='relu'),
#            tf.keras.layers.Flatten(),
#            tf.keras.layers.Dense(128, activation='relu'),
#            tf.keras.layers.Dense(10)
#        ])
#        model.compile(
#            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#            metrics=['accuracy'])
#        return model
#
#      per_worker_batch_size = 64
#      global_batch_size = per_worker_batch_size * ctx.num_workers
#      multi_worker_dataset = mnist_dataset(global_batch_size)
#
#      with strategy.scope():
#        multi_worker_model = build_and_compile_cnn_model()
#
#      multi_worker_model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=70)
#
#    with self.assertRaises(SystemExit):
#      cluster = TFCluster.run(self.sc, _map_fun, tf_args={}, num_executors=self.num_workers, num_ps=0, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief', release_port=False)
#      print("Expect hang in TensorFlow due to unreleased port.")
#      cluster.shutdown(timeout=10)


if __name__ == '__main__':
  unittest.main()

import numpy as np
import os
import shutil
import test
import unittest

from tensorflowonspark import compat
from tensorflowonspark.pipeline import HasBatchSize, HasSteps, Namespace, TFEstimator, TFParams
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class PipelineTest(test.SparkTest):
  @classmethod
  def setUpClass(cls):
    super(PipelineTest, cls).setUpClass()

    # create an artificial training dataset of two features with labels computed from known weights
    np.random.seed(1234)
    cls.features = np.random.rand(1000, 2)
    cls.weights = np.array([3.14, 1.618])
    cls.labels = np.matmul(cls.features, cls.weights)
    # convert to Python types for use with Spark DataFrames
    cls.train_examples = [(cls.features[i].tolist(), [cls.labels[i].item()]) for i in range(1000)]
    # create a simple test dataset
    cls.test_examples = [([1.0, 1.0], [0.0])]

    # define model_dir and export_dir for tests
    cls.model_dir = os.getcwd() + os.sep + "test_model"
    cls.export_dir = os.getcwd() + os.sep + "test_export"
    cls.tfrecord_dir = os.getcwd() + os.sep + "test_tfr"

  @classmethod
  def tearDownClass(cls):
    super(PipelineTest, cls).tearDownClass()

  def setUp(self):
    super(PipelineTest, self).setUp()
    # remove any prior test artifacts
    shutil.rmtree(self.model_dir, ignore_errors=True)
    shutil.rmtree(self.export_dir, ignore_errors=True)
    shutil.rmtree(self.tfrecord_dir, ignore_errors=True)

  def tearDown(self):
    # Note: don't clean up artifacts after test (in case we need to view/debug)
    pass

  def test_namespace(self):
    """Namespace class initializers"""
    # from dictionary
    d = {'string': 'foo', 'integer': 1, 'float': 3.14, 'array': [1, 2, 3], 'map': {'a': 1, 'b': 2}}
    n1 = Namespace(d)
    self.assertEqual(n1.string, 'foo')
    self.assertEqual(n1.integer, 1)
    self.assertEqual(n1.float, 3.14)
    self.assertEqual(n1.array, [1, 2, 3])
    self.assertEqual(n1.map, {'a': 1, 'b': 2})
    self.assertTrue('string' in n1)
    self.assertFalse('extra' in n1)

    # from namespace
    n2 = Namespace(n1)
    self.assertEqual(n2.string, 'foo')
    self.assertEqual(n2.integer, 1)
    self.assertEqual(n2.float, 3.14)
    self.assertEqual(n2.array, [1, 2, 3])
    self.assertEqual(n2.map, {'a': 1, 'b': 2})
    self.assertTrue('string' in n2)
    self.assertFalse('extra' in n2)

    # from argv list
    argv = ["--foo", "1", "--bar", "test", "--baz", "3.14"]
    n3 = Namespace(argv)
    self.assertEqual(n3.argv, argv)

  def test_TFParams(self):
    """Merging namespace args w/ ML Params"""
    class Foo(TFParams, HasBatchSize, HasSteps):
      def __init__(self, args):
        super(Foo, self).__init__()
        self.args = args

    n = Namespace({'a': 1, 'b': 2})
    f = Foo(n).setBatchSize(10).setSteps(100)
    combined_args = f.merge_args_params()
    expected_args = Namespace({'a': 1, 'b': 2, 'batch_size': 10, 'steps': 100})
    self.assertEqual(combined_args, expected_args)

  def test_spark_saved_model(self):
    """InputMode.SPARK TFEstimator w/ explicit saved_model export for TFModel inferencing"""

    def _spark_train(args, ctx):
      """Basic linear regression in a distributed TF cluster using InputMode.SPARK"""
      import tensorflow as tf
      from tensorflowonspark import TFNode

      tf.compat.v1.reset_default_graph()
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

      with strategy.scope():
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=[2]))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.2), loss='mse', metrics=['mse'])
        model.summary()

      tf_feed = TFNode.DataFeed(ctx.mgr, input_mapping=args.input_mapping)

      def rdd_generator():
        while not tf_feed.should_stop():
          batch = tf_feed.next_batch(1)
          if len(batch['x']) > 0:
            features = batch['x'][0]
            label = batch['y_'][0]
            yield (features, label)
          else:
            return

      ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (tf.TensorShape([2]), tf.TensorShape([1])))
      # disable auto-sharding since we're feeding from an RDD generator
      options = tf.data.Options()
      compat.disable_auto_shard(options)
      ds = ds.with_options(options)
      ds = ds.batch(args.batch_size)

      # only train 90% of each epoch to account for uneven RDD partition sizes
      steps_per_epoch = 1000 * 0.9 // (args.batch_size * ctx.num_workers)

      tf.io.gfile.makedirs(args.model_dir)
      filepath = args.model_dir + "/weights-{epoch:04d}"
      callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, load_weights_on_restart=True, save_weights_only=True)]

      model.fit(ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

      # This fails with: "NotImplementedError: `fit_generator` is not supported for models compiled with tf.distribute.Strategy"
      # model.fit_generator(ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

      if args.export_dir:
        print("exporting model to: {}".format(args.export_dir))
        compat.export_saved_model(model, args.export_dir, ctx.job_name == 'chief')

      tf_feed.terminate()

    # create a Spark DataFrame of training examples (features, labels)
    rdd = self.sc.parallelize(self.train_examples, 2)
    trainDF = rdd.toDF(['col1', 'col2'])

    # train and export model
    args = {}
    estimator = TFEstimator(_spark_train, args) \
                  .setInputMapping({'col1': 'x', 'col2': 'y_'}) \
                  .setModelDir(self.model_dir) \
                  .setExportDir(self.export_dir) \
                  .setClusterSize(self.num_workers) \
                  .setMasterNode("chief") \
                  .setNumPS(0) \
                  .setBatchSize(1) \
                  .setEpochs(1)
    model = estimator.fit(trainDF)
    self.assertTrue(os.path.isdir(self.export_dir))

    # create a Spark DataFrame of test examples (features, labels)
    testDF = self.spark.createDataFrame(self.test_examples, ['c1', 'c2'])

    # test saved_model using exported signature
    model.setTagSet('serve') \
          .setSignatureDefKey('serving_default') \
          .setInputMapping({'c1': 'dense_input'}) \
          .setOutputMapping({'dense': 'cout'})
    preds = model.transform(testDF).head()                  # take first/only result
    pred = preds.cout[0]                                    # unpack scalar from tensor
    expected = np.sum(self.weights)
    self.assertAlmostEqual(pred, expected, 2)

#  def test_spark_sparse_tensor(self):
#    """InputMode.SPARK feeding sparse tensors"""
#    def sparse_train(args, ctx):
#        import tensorflow as tf
#
#        # reset graph in case we're re-using a Spark python worker (during tests)
#        tf.compat.v1.reset_default_graph()
#
#        cluster, server = ctx.start_cluster_server(ctx)
#        if ctx.job_name == "ps":
#          server.join()
#        elif ctx.job_name == "worker":
#          with tf.device(tf.compat.v1.train.replica_device_setter(
#            worker_device="/job:worker/task:%d" % ctx.task_index,
#            cluster=cluster)):
#            y_ = tf.compat.v1.placeholder(tf.float32, name='y_label')
#            label = tf.identity(y_, name='label')
#
#            row_indices = tf.compat.v1.placeholder(tf.int64, name='x_row_indices')
#            col_indices = tf.compat.v1.placeholder(tf.int64, name='x_col_indices')
#            values = tf.compat.v1.placeholder(tf.float32, name='x_values')
#            indices = tf.stack([row_indices[0], col_indices[0]], axis=1)
#            data = values[0]
#
#            x = tf.SparseTensor(indices=indices, values=data, dense_shape=[args.batch_size, 10])
#            w = tf.Variable(tf.random.truncated_normal([10, 1]), name='w')
#            y = tf.sparse.sparse_dense_matmul(x, w, name='y')
#
#            global_step = tf.compat.v1.train.get_or_create_global_step()
#            cost = tf.reduce_mean(input_tensor=tf.square(y_ - y), name='cost')
#            optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cost, global_step)
#
#          with tf.compat.v1.train.MonitoredTrainingSession(master=server.target,
#                                                           is_chief=(ctx.task_index == 0),
#                                                           checkpoint_dir=args.model_dir,
#                                                           save_checkpoint_steps=20) as sess:
#            tf_feed = ctx.get_data_feed(input_mapping=args.input_mapping)
#            while not sess.should_stop() and not tf_feed.should_stop():
#              batch = tf_feed.next_batch(args.batch_size)
#              if len(batch) > 0:
#                print("batch: {}".format(batch))
#                feed = {y_: batch['y_label'],
#                        row_indices: batch['x_row_indices'],
#                        col_indices: batch['x_col_indices'],
#                        values: batch['x_values']}
#                _, pred, trained_weights = sess.run([optimizer, y, w], feed_dict=feed)
#                print("trained_weights: {}".format(trained_weights))
#            sess.close()
#
#          # wait for MonitoredTrainingSession to save last checkpoint
#          time.sleep(10)
#
#    args = {}
#    estimator = TFEstimator(sparse_train, args) \
#              .setInputMapping({'labels': 'y_label', 'row_indices': 'x_row_indices', 'col_indices': 'x_col_indices', 'values': 'x_values'}) \
#              .setInputMode(TFCluster.InputMode.SPARK) \
#              .setModelDir(self.model_dir) \
#              .setClusterSize(self.num_workers) \
#              .setNumPS(1) \
#              .setBatchSize(1)
#
#    model_weights = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]).T
#    examples = [scipy.sparse.random(1, 10, density=0.5,) for i in range(200)]
#    rdd = self.sc.parallelize(examples).map(lambda e: ((e * model_weights).tolist()[0][0], e.row.tolist(), e.col.tolist(), e.data.tolist()))
#    df = rdd.toDF(["labels", "row_indices", "col_indices", "values"])
#    df.show(5)
#    model = estimator.fit(df)
#
#    model.setOutputMapping({'label': 'label', 'y/SparseTensorDenseMatMul': 'predictions'})
#    test_examples = [scipy.sparse.random(1, 10, density=0.5,) for i in range(50)]
#    test_rdd = self.sc.parallelize(test_examples).map(lambda e: ((e * model_weights).tolist()[0][0], e.row.tolist(), e.col.tolist(), e.data.tolist()))
#    test_df = test_rdd.toDF(["labels", "row_indices", "col_indices", "values"])
#    test_df.show(5)
#    preds = model.transform(test_df)
#    preds.show(5)


if __name__ == '__main__':
  unittest.main()

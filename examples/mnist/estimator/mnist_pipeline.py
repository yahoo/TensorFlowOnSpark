# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_estimator

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  import numpy as np
  import tensorflow as tf
  import tensorflow_datasets as tfds
  from tensorflowonspark import TFNode

  tfds.disable_progress_bar()

  class StopFeedHook(tf.estimator.SessionRunHook):
    """SessionRunHook to terminate InputMode.SPARK RDD feeding if the training loop exits before the entire RDD is consumed."""

    def __init__(self, feed):
      self.feed = feed

    def end(self, session):
      self.feed.terminate()
      self.feed.next_batch(1)

  BATCH_SIZE = args.batch_size
  LEARNING_RATE = args.learning_rate

  tf_feed = TFNode.DataFeed(ctx.mgr)

  def rdd_generator():
    while not tf_feed.should_stop():
      batch = tf_feed.next_batch(1)
      if len(batch) > 0:
        example = batch[0]
        image = np.array(example[0]).astype(np.float32) / 255.0
        image = np.reshape(image, (28, 28, 1))
        label = np.array(example[1]).astype(np.float32)
        label = np.reshape(label, (1,))
        yield (image, label)
      else:
        return

  def input_fn(mode, input_context=None):
    if mode == tf.estimator.ModeKeys.TRAIN:
      # Note: Spark is responsible for sharding/repeating/shuffling the data via RDD
      ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
      return ds.batch(BATCH_SIZE)
    else:
      # read evaluation data from tensorflow_datasets directly
      def scale(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

      mnist = tfds.load(name='mnist', with_info=True, as_supervised=True)
      ds = mnist['test']
      if input_context:
        ds = ds.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
      return ds.map(scale).batch(BATCH_SIZE)

  def serving_input_receiver_fn():
    features = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='features')
    receiver_tensors = {'features': features}
    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

  def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {'logits': logits}
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(input_tensor=loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=100)

  classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=args.model_dir, config=config)

  # exporter = tf.estimator.FinalExporter("serving", serving_input_receiver_fn=serving_input_receiver_fn)

  # Note: MultiWorkerMirroredStrategy (CollectiveAllReduceStrategy) is synchronous,
  # so we need to ensure that all workers complete training before any of them run out of data from the RDD.
  # And given that Spark RDD partitions (and partition sizes) can be non-evenly divisible by num_workers,
  # we'll just stop training at 90% of the total expected number of steps.
  steps = 60000 * args.epochs / args.batch_size
  steps_per_worker = steps / ctx.num_workers
  max_steps_per_worker = steps_per_worker * 0.9

  tf.estimator.train_and_evaluate(
      classifier,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=max_steps_per_worker, hooks=[StopFeedHook(tf_feed)]),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
      # eval_spec=tf.estimator.EvalSpec(input_fn=input_fn, exporters=exporter)
  )

  if ctx.job_name == 'chief':
    print("Exporting saved_model to {}".format(args.export_dir))
    classifier.export_saved_model(args.export_dir, serving_input_receiver_fn)


if __name__ == "__main__":

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import udf
  from pyspark.sql.types import IntegerType
  from tensorflowonspark import dfutil
  from tensorflowonspark.pipeline import TFEstimator, TFModel
  import argparse

  sc = SparkContext(conf=SparkConf().setAppName("mnist_estimator"))
  spark = SparkSession(sc)

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=10000)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
  parser.add_argument("--format", help="example format: (csv|tfr)", choices=["csv", "tfr"], default="csv")
  parser.add_argument("--images_labels", help="path to MNIST images and labels in parallelized format")
  parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
  parser.add_argument("--mode", help="train|inference", choices=["train", "inference"], default="train")
  parser.add_argument("--model_dir", help="path to save checkpoint", default="mnist_model")
  parser.add_argument("--export_dir", help="path to export saved_model", default="mnist_export")
  parser.add_argument("--output", help="HDFS path to save predictions", type=str, default="predictions")
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  args = parser.parse_args()
  print("args:", args)

  if args.format == 'tfr':
    # load TFRecords as a DataFrame
    df = dfutil.loadTFRecords(sc, args.images_labels)
  else:  # args.format == 'csv':
    # create RDD of input data
    def parse(ln):
      vec = [int(x) for x in ln.split(',')]
      return (vec[1:], vec[0])

    images_labels = sc.textFile(args.images_labels).map(parse)
    df = spark.createDataFrame(images_labels, ['image', 'label'])

  df.show()

  if args.mode == 'train':
    estimator = TFEstimator(main_fun, args) \
        .setInputMapping({'image': 'image', 'label': 'label'}) \
        .setModelDir(args.model_dir) \
        .setExportDir(args.export_dir) \
        .setClusterSize(args.cluster_size) \
        .setTensorboard(args.tensorboard) \
        .setEpochs(args.epochs) \
        .setBatchSize(args.batch_size) \
        .setGraceSecs(60)
    model = estimator.fit(df)
  else:  # args.mode == 'inference':
    # using a trained/exported model
    model = TFModel(args) \
        .setInputMapping({'image': 'features'}) \
        .setOutputMapping({'logits': 'prediction'}) \
        .setExportDir(args.export_dir) \
        .setBatchSize(args.batch_size)

    def argmax_fn(l):
      return max(range(len(l)), key=lambda i: l[i])

    argmax = udf(argmax_fn, IntegerType())

    preds = model.transform(df).withColumn('argmax', argmax('prediction'))
    preds.show()
    preds.write.json(args.output)

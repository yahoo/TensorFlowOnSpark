# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  import numpy as np
  import tensorflow as tf
  from tensorflowonspark import compat, TFNode

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

  # single node
  # single_worker_model = build_and_compile_cnn_model()
  # single_worker_model.fit(x=train_datasets, epochs=3)

  tf_feed = TFNode.DataFeed(ctx.mgr, False)

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

  ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
  ds = ds.batch(args.batch_size)

  # this fails
  # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=args.model_dir)]
  tf.io.gfile.makedirs(args.model_dir)
  filepath = args.model_dir + "/weights-{epoch:04d}"
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)]

  with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

  # Note: MultiWorkerMirroredStrategy (CollectiveAllReduceStrategy) is synchronous,
  # so we need to ensure that all workers complete training before any of them run out of data from the RDD.
  # And given that Spark RDD partitions (and partition sizes) can be non-evenly divisible by num_workers,
  # we'll just stop training at 90% of the total expected number of steps.
  steps_per_epoch = 60000 / args.batch_size
  steps_per_epoch_per_worker = steps_per_epoch / ctx.num_workers
  max_steps_per_worker = steps_per_epoch_per_worker * 0.9

  multi_worker_model.fit(x=ds, epochs=args.epochs, steps_per_epoch=max_steps_per_worker, callbacks=callbacks)

  from tensorflow_estimator.python.estimator.export import export_lib
  export_dir = export_lib.get_timestamped_export_dir(args.export_dir)
  compat.export_saved_model(multi_worker_model, export_dir, ctx.job_name == 'chief')

  # terminating feed tells spark to skip processing further partitions
  tf_feed.terminate()


if __name__ == '__main__':
  import argparse
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import udf
  from pyspark.sql.types import IntegerType
  from tensorflowonspark import dfutil
  from tensorflowonspark.pipeline import TFEstimator, TFModel

  sc = SparkContext(conf=SparkConf().setAppName("mnist_keras"))
  spark = SparkSession(sc)

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
  parser.add_argument("--format", help="example format: (csv|tfr)", choices=["csv", "tfr"], default="csv")
  parser.add_argument("--images_labels", help="path to MNIST images and labels in parallelized format")
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
        .setInputMapping({'image': 'conv2d_input'}) \
        .setOutputMapping({'dense_1': 'prediction'}) \
        .setExportDir(args.export_dir) \
        .setBatchSize(args.batch_size)

    def argmax_fn(l):
      return max(range(len(l)), key=lambda i: l[i])

    argmax = udf(argmax_fn, IntegerType())

    preds = model.transform(df).withColumn('argmax', argmax('prediction'))
    preds.show()
    preds.write.json(args.output)

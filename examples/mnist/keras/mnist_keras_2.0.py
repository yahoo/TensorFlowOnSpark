# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  import tensorflow_datasets as tfds
  import tensorflow as tf
  tfds.disable_progress_bar()

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  BUFFER_SIZE = args.buffer_size
  BATCH_SIZE = args.batch_size
  NUM_WORKERS = args.cluster_size

  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                             with_info=True,
                             as_supervised=True)

  train_datasets_unbatched = datasets['train'].repeat().map(scale).shuffle(BUFFER_SIZE)

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

  # Here the batch size scales up by number of workers since
  # `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
  # and now this becomes 128.
  GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
  train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
  with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

  if ctx.job_name == 'chief':
    # multi_worker_model.save(args.model_dir, save_format='tf')
    tf.keras.experimental.export_saved_model(multi_worker_model, args.model_dir)


if __name__ == '__main__':
  import argparse
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFCluster

  sc = SparkContext(conf=SparkConf().setAppName("mnist_keras"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=10000)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs of training data", type=int, default=5)
  parser.add_argument("--model_dir", help="path to save model/checkpoint", default="mnist_model")
  parser.add_argument("--steps_per_epoch", help="number of steps per epoch", type=int, default=469)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  args = parser.parse_args()
  print("args:", args)

  cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps=0, tensorboard=args.tensorboard, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
  cluster.shutdown()

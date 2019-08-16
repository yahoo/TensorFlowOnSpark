# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_estimator

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  import tensorflow_datasets as tfds
  import tensorflow as tf

  BUFFER_SIZE = args.buffer_size
  BATCH_SIZE = args.batch_size
  LEARNING_RATE = args.learning_rate

  def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                                  with_info=True,
                                  as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                     datasets['test'])

    def scale(image, label):
      image = tf.cast(image, tf.float32)
      image /= 255
      return image, label

    if input_context:
      mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                          input_context.input_pipeline_id)
    return mnist_dataset.repeat(args.epochs).map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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

  tf.estimator.train_and_evaluate(
      classifier,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
      # eval_spec=tf.estimator.EvalSpec(input_fn=input_fn, exporters=exporter)
  )

  if ctx.job_name == 'chief':
    print("========== exporting saved_model to {}".format(args.export_dir))
    classifier.export_saved_model(args.export_dir, serving_input_receiver_fn)


if __name__ == "__main__":
  # tf.app.run()

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFCluster
  import argparse

  sc = SparkContext(conf=SparkConf().setAppName("mnist_estimator"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=10000)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
  parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-4)
  parser.add_argument("--model_dir", help="path to save checkpoint", default="mnist_model")
  parser.add_argument("--export_dir", help="path to export saved_model", default="mnist_export")
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  args = parser.parse_args()
  print("args:", args)

  cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps=0, tensorboard=args.tensorboard, input_mode=TFCluster.InputMode.TENSORFLOW, log_dir=args.model_dir, master_node='chief', eval_node=True)
  cluster.shutdown(grace_secs=120)

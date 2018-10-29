'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function


def main_fun(args, ctx):
  import numpy
  import os
  import tensorflow as tf
  from tensorflow.python import keras
  from tensorflow.python.keras import backend as K
  from tensorflow.python.keras.datasets import mnist
  from tensorflow.python.keras.models import Sequential, load_model, save_model
  from tensorflow.python.keras.layers import Dense, Dropout
  from tensorflow.python.keras.optimizers import RMSprop
  from tensorflow.python.keras.callbacks import LambdaCallback, TensorBoard
  from tensorflow.python.saved_model import builder as saved_model_builder
  from tensorflow.python.saved_model import tag_constants
  from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
  from tensorflowonspark import TFNode

  cluster, server = TFNode.start_cluster_server(ctx)

  if ctx.job_name == "ps":
    server.join()
  elif ctx.job_name == "worker":

    def generate_rdd_data(tf_feed, batch_size):
        print("generate_rdd_data invoked")
        while True:
            batch = tf_feed.next_batch(batch_size)
            imgs = []
            lbls = []
            for item in batch:
                imgs.append(item[0])
                lbls.append(item[1])
            images = numpy.array(imgs).astype('float32') / 255
            labels = numpy.array(lbls).astype('float32')
            yield (images, labels)

    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % ctx.task_index,
      cluster=cluster)):

      IMAGE_PIXELS = 28
      batch_size = 100
      num_classes = 10

      # the data, shuffled and split between train and test sets
      if args.input_mode == 'tf':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
      else:  # args.mode == 'spark'
        x_train = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x_train")
        y_train = tf.placeholder(tf.float32, [None, 10], name="y_train")
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(10000, 784)
        y_test = keras.utils.to_categorical(y_test, num_classes)

      model = Sequential()
      model.add(Dense(512, activation='relu', input_shape=(784,)))
      model.add(Dropout(0.2))
      model.add(Dense(512, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(10, activation='softmax'))

      model.summary()

      model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])

    saver = tf.train.Saver()

    with tf.Session(server.target) as sess:
      K.set_session(sess)

      def save_checkpoint(epoch, logs=None):
        if epoch == 1:
          tf.train.write_graph(sess.graph.as_graph_def(), args.model_dir, 'graph.pbtxt')
        saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), global_step=epoch * args.steps_per_epoch)

      ckpt_callback = LambdaCallback(on_epoch_end=save_checkpoint)
      tb_callback = TensorBoard(log_dir=args.model_dir, histogram_freq=1, write_graph=True, write_images=True)

      # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
      callbacks = [ckpt_callback, tb_callback] if ctx.task_index == 0 else None

      if args.input_mode == 'tf':
        # train & validate on in-memory data
        model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=args.epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)
      else:  # args.input_mode == 'spark':
        # train on data read from a generator which is producing data from a Spark RDD
        tf_feed = TFNode.DataFeed(ctx.mgr)
        model.fit_generator(generator=generate_rdd_data(tf_feed, batch_size),
                            steps_per_epoch=args.steps_per_epoch,
                            epochs=args.epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)

      if args.export_dir and ctx.job_name == 'worker' and ctx.task_index == 0:
        # save a local Keras model, so we can reload it with an inferencing learning_phase
        save_model(model, "tmp_model")

        # reload the model
        K.set_learning_phase(False)
        new_model = load_model("tmp_model")

        # export a saved_model for inferencing
        builder = saved_model_builder.SavedModelBuilder(args.export_dir)
        signature = predict_signature_def(inputs={'images': new_model.input},
                                          outputs={'scores': new_model.output})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature},
                                             clear_devices=True)
        builder.save()

      if args.input_mode == 'spark':
        tf_feed.terminate()


if __name__ == '__main__':
    import argparse
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from tensorflowonspark import TFCluster

    sc = SparkContext(conf=SparkConf().setAppName("mnist_mlp"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    num_ps = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--epochs", help="number of epochs of training data", type=int, default=20)
    parser.add_argument("--export_dir", help="directory to export saved_model")
    parser.add_argument("--images", help="HDFS path to MNIST images in parallelized CSV format")
    parser.add_argument("--input_mode", help="input mode (tf|spark)", default="tf")
    parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized CSV format")
    parser.add_argument("--model_dir", help="directory to write model checkpoints")
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--steps_per_epoch", help="number of steps per epoch", type=int, default=300)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    if args.input_mode == 'tf':
      cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW, log_dir=args.model_dir)
    else:  # args.input_mode == 'spark':
      cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.SPARK, log_dir=args.model_dir)
      images = sc.textFile(args.images).map(lambda ln: [float(x) for x in ln.split(',')])
      labels = sc.textFile(args.labels).map(lambda ln: [float(x) for x in ln.split(',')])
      dataRDD = images.zip(labels)
      cluster.train(dataRDD, args.epochs)

    cluster.shutdown()

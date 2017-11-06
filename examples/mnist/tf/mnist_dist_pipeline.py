# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflowonspark import TFNode
from datetime import datetime
import logging
import math
import os
import tensorflow as tf

# Parameters
hidden_units = 128
batch_size   = 100
IMAGE_PIXELS = 28

def map_fun(args, ctx):
  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.protocol == 'rdma')

  if job_name == "ps":
    server.join()
  elif job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

      # Variables of the hidden layer
      hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
      tf.summary.histogram("hidden_weights", hid_w)

      # Variables of the softmax layer
      sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                              stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
      tf.summary.histogram("softmax_weights", sm_w)

      # Placeholders or QueueRunner/Readers for input data
      num_epochs = None if args.epochs == 0 else args.epochs
      images = TFNode.hdfs_path(ctx, args.tfrecord_dir)
      x, y_ = read_tfr_examples(images, 100, num_epochs)

      x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
      tf.summary.image("x_img", x_img)

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

      global_step = tf.Variable(0)

      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      tf.summary.scalar("loss", loss)
      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      # Test trained model
      label = tf.argmax(y_, 1, name="label")
      prediction = tf.argmax(y, 1,name="prediction")
      correct_prediction = tf.equal(prediction, label)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model_dir)
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" % (worker_num), graph=tf.get_default_graph())

    sv = tf.train.Supervisor(is_chief=(task_index == 0),
                             logdir=logdir,
                             init_op=init_op,
                             summary_op=None,
                             saver=saver,
                             global_step=global_step,
                             stop_grace_secs=300,
                             save_model_secs=10)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      print("{0} session ready".format(datetime.now().isoformat()))

      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < args.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using QueueRunners/Readers
        if (step % 100 == 0):
          print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
        _, summary, step = sess.run([train_op, summary_op, global_step])
        if sv.is_chief:
          summary_writer.add_summary(summary, step)

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()

def export_fun(args):
  """Define/export a single-node TF graph for inferencing"""
  # Input placeholder for inferencing
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")

  # Variables of the hidden layer
  hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                          stddev=1.0 / IMAGE_PIXELS), name="hid_w")
  hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
  tf.summary.histogram("hidden_weights", hid_w)

  # Variables of the softmax layer
  sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                          stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)
  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
  prediction = tf.argmax(y, 1, name="prediction")

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # load graph from a checkpoint
    logging.info("model path: {}".format(args.model_dir))
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    logging.info("ckpt: {}".format(ckpt))
    assert ckpt and ckpt.model_checkpoint_path, "Invalid model checkpoint path: {}".format(args.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    logging.info("Exporting saved_model to: {}".format(args.export_dir))
    # exported signatures defined in code
    signatures = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
        'inputs': { 'image': x },
        'outputs': { 'prediction': prediction },
        'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      },
      'featurize': {
        'inputs': { 'image': x },
        'outputs': { 'features': hid },
        'method_name': 'featurize'
      }
    }
    TFNode.export_saved_model(sess,
                              args.export_dir,
                              tf.saved_model.tag_constants.SERVING,
                              signatures)
    logging.info("Exported saved_model")


def read_csv_examples(image_dir, label_dir, batch_size=100, num_readers=1, num_epochs=None, task_index=None, num_workers=None):
  logging.info("num_epochs: {0}".format(num_epochs))
  # Setup queue of csv image filenames
  tf_record_pattern = os.path.join(image_dir, 'part-*')
  images = tf.gfile.Glob(tf_record_pattern)
  logging.info("images: {0}".format(images))
  image_queue = tf.train.string_input_producer(images, shuffle=False, capacity=1000, num_epochs=num_epochs, name="image_queue")

  # Setup queue of csv label filenames
  tf_record_pattern = os.path.join(label_dir, 'part-*')
  labels = tf.gfile.Glob(tf_record_pattern)
  logging.info("labels: {0}".format(labels))
  label_queue = tf.train.string_input_producer(labels, shuffle=False, capacity=1000, num_epochs=num_epochs, name="label_queue")

  # Setup reader for image queue
  img_reader = tf.TextLineReader(name="img_reader")
  _, img_csv = img_reader.read(image_queue)
  image_defaults = [ [1.0] for col in range(784) ]
  img = tf.stack(tf.decode_csv(img_csv, image_defaults))
  # Normalize values to [0,1]
  norm = tf.constant(255, dtype=tf.float32, shape=(784,))
  image = tf.div(img, norm)
  logging.info("image: {0}".format(image))

  # Setup reader for label queue
  label_reader = tf.TextLineReader(name="label_reader")
  _, label_csv = label_reader.read(label_queue)
  label_defaults = [ [1.0] for col in range(10) ]
  label = tf.stack(tf.decode_csv(label_csv, label_defaults))
  logging.info("label: {0}".format(label))

  # Return a batch of examples
  return tf.train.batch([image,label], batch_size, num_threads=num_readers, name="batch_csv")

def read_tfr_examples(path, batch_size=100, num_epochs=None, num_readers=1, task_index=None, num_workers=None):
  logging.info("num_epochs: {0}".format(num_epochs))

  # Setup queue of TFRecord filenames
  tf_record_pattern = os.path.join(path, 'part-*')
  files = tf.gfile.Glob(tf_record_pattern)
  queue_name = "file_queue"

  # split input files across workers, if specified
  if task_index is not None and num_workers is not None:
    num_files = len(files)
    files = files[task_index:num_files:num_workers]
    queue_name = "file_queue_{0}".format(task_index)

  logging.info("files: {0}".format(files))
  file_queue = tf.train.string_input_producer(files, shuffle=False, capacity=1000, num_epochs=num_epochs, name=queue_name)

  # Setup reader for examples
  reader = tf.TFRecordReader(name="reader")
  _, serialized = reader.read(file_queue)
  feature_def = {'label': tf.FixedLenFeature([10], tf.int64), 'image': tf.FixedLenFeature([784], tf.int64) }
  features = tf.parse_single_example(serialized, feature_def)
  norm = tf.constant(255, dtype=tf.float32, shape=(784,))
  image = tf.div(tf.to_float(features['image']), norm)
  logging.info("image: {0}".format(image))
  label = tf.to_float(features['label'])
  logging.info("label: {0}".format(label))

  # Return a batch of examples
  return tf.train.batch([image,label], batch_size, num_threads=num_readers, name="batch")

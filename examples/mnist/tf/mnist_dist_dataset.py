# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def print_log(worker_num, arg):
  print("%d: " % worker_num, end=" ")
  print(arg)


def map_fun(args, ctx):
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import os
  import tensorflow as tf
  import time

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Parameters
  IMAGE_PIXELS = 28
  hidden_units = 128

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

  def _parse_csv(ln):
    splits = tf.string_split([ln], delimiter='|')
    lbl = splits.values[0]
    img = splits.values[1]
    image_defaults = [[0.0] for col in range(IMAGE_PIXELS * IMAGE_PIXELS)]
    image = tf.stack(tf.decode_csv(img, record_defaults=image_defaults))
    norm = tf.constant(255, dtype=tf.float32, shape=(784,))
    normalized_image = tf.div(image, norm)
    label_value = tf.string_to_number(lbl, tf.int32)
    label = tf.one_hot(label_value, 10)
    return (normalized_image, label, label_value)

  def _parse_tfr(example_proto):
    print("example_proto: {}".format(example_proto))
    feature_def = {"label": tf.FixedLenFeature(10, tf.int64),
                   "image": tf.FixedLenFeature(IMAGE_PIXELS * IMAGE_PIXELS, tf.int64)}
    features = tf.parse_single_example(example_proto, feature_def)
    norm = tf.constant(255, dtype=tf.float32, shape=(784,))
    image = tf.div(tf.to_float(features['image']), norm)
    label = tf.to_float(features['label'])
    return (image, label)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      cluster=cluster)):

      # Dataset for input data
      image_dir = TFNode.hdfs_path(ctx, args.images)
      file_pattern = os.path.join(image_dir, 'part-*')
      files = tf.gfile.Glob(file_pattern)

      parse_fn = _parse_tfr if args.format == 'tfr' else _parse_csv
      ds = tf.data.TextLineDataset(files).map(parse_fn).batch(args.batch_size)
      iterator = ds.make_initializable_iterator()
      x, y_, y_val = iterator.get_next()

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
      prediction = tf.argmax(y, 1, name="prediction")
      correct_prediction = tf.equal(prediction, label)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model)
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" % worker_num, graph=tf.get_default_graph())

    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=10)
    else:
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=0)
      output_dir = TFNode.hdfs_path(ctx, args.output)
      tf.gfile.MkDir(output_dir)
      output_file = tf.gfile.Open("{0}/part-{1:05d}".format(output_dir, worker_num), mode='w')

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      print("{0} session ready".format(datetime.now().isoformat()))

      # Loop until the supervisor shuts down or 1000000 steps have completed.
      sess.run(iterator.initializer)
      step = 0
      count = 0
      while not sv.should_stop() and step < args.steps:

        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using QueueRunners/Readers
        if args.mode == "train":
          if (step % 100 == 0):
            print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
          _, summary, step, yv = sess.run([train_op, summary_op, global_step, y_val])
          # print("yval: {}".format(yv))
          if sv.is_chief:
            summary_writer.add_summary(summary, step)
        else:  # args.mode == "inference"
          labels, pred, acc = sess.run([label, prediction, accuracy])
          # print("label: {0}, pred: {1}".format(labels, pred))
          print("acc: {0}".format(acc))
          for i in range(len(labels)):
            count += 1
            output_file.write("{0} {1}\n".format(labels[i], pred[i]))
          print("count: {0}".format(count))

    if args.mode == "inference":
      output_file.close()
      # Delay chief worker from shutting down supervisor during inference, since it can load model, start session,
      # run inference and request stop before the other workers even start/sync their sessions.
      if task_index == 0:
        time.sleep(60)

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()

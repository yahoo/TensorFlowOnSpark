# Copyright 2018 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function


def print_log(worker_num, arg):
  print("{0}: {1}".format(worker_num, arg))


def map_fun(args, ctx):
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  IMAGE_PIXELS = 28
  hidden_units = 128
  batch_size = args.batch_size

  # Get TF cluster and server instances
  cluster, server = ctx.start_cluster_server(1, args.rdma)

  def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    xs = xs / 255.0
    ys = numpy.array(labels)
    ys = ys.astype(numpy.uint8)
    return (xs, ys)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      cluster=cluster)):

      # Placeholders or QueueRunner/Readers for input data
      with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

        x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
        tf.summary.image("x_img", x_img)

      with tf.name_scope('layer'):
        # Variables of the hidden layer
        with tf.name_scope('hidden_layer'):
          hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units], stddev=1.0 / IMAGE_PIXELS), name="hid_w")
          hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
          tf.summary.histogram("hidden_weights", hid_w)
          hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
          hid = tf.nn.relu(hid_lin)

        # Variables of the softmax layer
        with tf.name_scope('softmax_layer'):
          sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10], stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
          sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
          tf.summary.histogram("softmax_weights", sm_w)
          y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

      global_step = tf.train.get_or_create_global_step()

      with tf.name_scope('loss'):
        loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        tf.summary.scalar("loss", loss)

      with tf.name_scope('train'):
        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

      # Test trained model
      label = tf.argmax(y_, 1, name="label")
      prediction = tf.argmax(y, 1, name="prediction")
      correct_prediction = tf.equal(prediction, label)

      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)

      summary_op = tf.summary.merge_all()

    logdir = ctx.absolute_path(args.model)
    print("tensorflow model path: {0}".format(logdir))
    hooks = [tf.train.StopAtStepHook(last_step=100000)]

    if job_name == "worker" and task_index == 0:
      summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # The MonitoredTrainingSession takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs
    with tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(task_index == 0),
                                             checkpoint_dir=logdir,
                                             hooks=hooks) as mon_sess:

      step = 0
      tf_feed = ctx.get_data_feed(args.mode == "train")
      while not mon_sess.should_stop() and not tf_feed.should_stop() and step < args.steps:
        # Run a training step asynchronously
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys}

        if len(batch_xs) > 0:
          if args.mode == "train":
            _, summary, step = mon_sess.run([train_op, summary_op, global_step], feed_dict=feed)
            # print accuracy and save model checkpoint to HDFS every 100 steps
            if (step % 100 == 0):
              print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, mon_sess.run(accuracy, {x: batch_xs, y_: batch_ys})))

            if task_index == 0:
              summary_writer.add_summary(summary, step)
          else:  # args.mode == "inference"
            labels, preds, acc = mon_sess.run([label, prediction, accuracy], feed_dict=feed)

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l, p in zip(labels, preds)]
            tf_feed.batch_results(results)
            print("results: {0}, acc: {1}".format(results, acc))

      if mon_sess.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

  if job_name == "worker" and task_index == 0:
    summary_writer.close()

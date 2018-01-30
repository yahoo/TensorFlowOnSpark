# Copyright 2017 Yahoo Inc.
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
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  IMAGE_PIXELS = 28

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  hidden_units = 128
  batch_size = args.batch_size

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(
      ctx, 1, args.protocol == 'rdma')

  def feed_dict(batch):
    # Convert from dict of named arrays to two numpy arrays of the proper type
    images = batch['image']
    labels = batch['label']
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
      x = tf.placeholder(
          tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
      y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

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

      accuracy = tf.reduce_mean(
          tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model_dir)
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter(
        "tensorboard_%d" % (worker_num), graph=tf.get_default_graph())

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
      #tf_feed = TFNode.DataFeed(ctx.mgr)
      tf_feed = TFNode.DataFeed(
          ctx.mgr, input_mapping=args.input_mapping)
      while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys}

        if len(batch_xs) > 0:
          _, summary, step = sess.run(
              [train_op, summary_op, global_step], feed_dict=feed)
          # print accuracy and save model checkpoint to HDFS every 100 steps
          if (step % 100 == 0):
            print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(
            ), step, sess.run(accuracy, {x: batch_xs, y_: batch_ys})))

          if sv.is_chief:
            summary_writer.add_summary(summary, step)

      if sv.should_stop() or step >= args.steps:
        tf_feed.terminate()

      if sv.is_chief and args.export_dir:
        print("{0} exporting saved_model to: {1}".format(
            datetime.now().isoformat(), args.export_dir))
        # exported signatures defined in code
        signatures = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
                'inputs': {'image': x},
                'outputs': {'prediction': prediction},
                'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            },
            'featurize': {
                'inputs': {'image': x},
                'outputs': {'features': hid},
                'method_name': 'featurize'
            }
        }
        TFNode.export_saved_model(sess,
                                  args.export_dir,
                                  tf.saved_model.tag_constants.SERVING,
                                  signatures)
      else:
        # non-chief workers should wait for chief
        while not sv.should_stop():
          print("Waiting for chief")
          time.sleep(5)

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()

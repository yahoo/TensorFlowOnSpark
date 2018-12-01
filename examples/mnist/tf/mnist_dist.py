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
  from datetime import datetime
  from tensorflowonspark import TFNode
  import math
  import os
  import tensorflow as tf
  import time

  num_workers = len(ctx.cluster_spec['worker'])
  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Parameters
  IMAGE_PIXELS = 28
  hidden_units = 128

  # Get TF cluster and server instances
  cluster, server = ctx.start_cluster_server(1, args.rdma)

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
    return (normalized_image, label)

  def _parse_tfr(example_proto):
    feature_def = {"label": tf.FixedLenFeature(10, tf.int64),
                   "image": tf.FixedLenFeature(IMAGE_PIXELS * IMAGE_PIXELS, tf.int64)}
    features = tf.parse_single_example(example_proto, feature_def)
    norm = tf.constant(255, dtype=tf.float32, shape=(784,))
    image = tf.div(tf.to_float(features['image']), norm)
    label = tf.to_float(features['label'])
    return (image, label)

  def build_model(graph, x):
    with graph.as_default():
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

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      prediction = tf.argmax(y, 1, name="prediction")
      return y, prediction

  if job_name == "ps":
    server.join()
  elif job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      cluster=cluster)):

      # Dataset for input data
      image_dir = ctx.absolute_path(args.images_labels)
      file_pattern = os.path.join(image_dir, 'part-*')

      ds = tf.data.Dataset.list_files(file_pattern)
      ds = ds.shard(num_workers, task_index).repeat(args.epochs).shuffle(args.shuffle_size)
      if args.format == 'csv2':
        ds = ds.interleave(tf.data.TextLineDataset, cycle_length=args.readers, block_length=1)
        parse_fn = _parse_csv
      else:  # args.format == 'tfr'
        ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=args.readers, block_length=1)
        parse_fn = _parse_tfr
      ds = ds.map(parse_fn).batch(args.batch_size)
      iterator = ds.make_one_shot_iterator()
      x, y_ = iterator.get_next()

      # Build core model
      y, prediction = build_model(tf.get_default_graph(), x)

      # Add training bits
      x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
      tf.summary.image("x_img", x_img)

      global_step = tf.train.get_or_create_global_step()

      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      tf.summary.scalar("loss", loss)
      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      label = tf.argmax(y_, 1, name="label")
      correct_prediction = tf.equal(prediction, label)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      tf.summary.scalar("acc", accuracy)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    model_dir = ctx.absolute_path(args.model)
    export_dir = ctx.absolute_path(args.export)
    print("tensorflow model path: {0}".format(model_dir))
    print("tensorflow export path: {0}".format(export_dir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" % worker_num, graph=tf.get_default_graph())

    if args.mode == 'inference':
      output_dir = ctx.absolute_path(args.output)
      print("output_dir: {}".format(output_dir))
      tf.gfile.MkDir(output_dir)
      output_file = tf.gfile.Open("{}/part-{:05d}".format(output_dir, task_index), mode='w')

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),
                                           checkpoint_dir=model_dir,
                                           hooks=[tf.train.StopAtStepHook(last_step=args.steps)]) as sess:
      print("{} session ready".format(datetime.now().isoformat()))

      # Loop until the session shuts down
      step = 0
      count = 0
      while not sess.should_stop():

        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        if args.mode == "train":
          if (step % 100 == 0):
            print("{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
          _, summary, step = sess.run([train_op, summary_op, global_step])
          if task_index == 0:
            summary_writer.add_summary(summary, step)
        else:  # args.mode == "inference"
          labels, pred, acc = sess.run([label, prediction, accuracy])
          # print("label: {0}, pred: {1}".format(labels, pred))
          print("acc: {}".format(acc))
          for i in range(len(labels)):
            count += 1
            output_file.write("{} {}\n".format(labels[i], pred[i]))
          print("count: {}".format(count))

    if args.mode == 'inference':
      output_file.close()

    print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

    # export model (on chief worker only)
    if args.mode == "train" and task_index == 0:
      tf.reset_default_graph()

      # add placeholders for input images (and optional labels)
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name='x')
      y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
      label = tf.argmax(y_, 1, name="label")

      # add core model
      y, prediction = build_model(tf.get_default_graph(), x)

      # restore from last checkpoint
      saver = tf.train.Saver()
      with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        print("ckpt: {}".format(ckpt))
        assert ckpt, "Invalid model checkpoint path: {}".format(model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        print("Exporting saved_model to: {}".format(export_dir))
        # exported signatures defined in code
        signatures = {
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
            'inputs': { 'image': x },
            'outputs': { 'prediction': prediction },
            'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          }
        }
        TFNode.export_saved_model(sess,
                                  export_dir,
                                  tf.saved_model.tag_constants.SERVING,
                                  signatures)
        print("Exported saved_model")

    # WORKAROUND for https://github.com/tensorflow/tensorflow/issues/21745
    # wait for all other nodes to complete (via done files)
    done_dir = "{}/{}/done".format(ctx.absolute_path(args.model), args.mode)
    print("Writing done file to: {}".format(done_dir))
    tf.gfile.MakeDirs(done_dir)
    with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
      done_file.write("done")

    for i in range(60):
      if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
        print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
        time.sleep(1)
      else:
        print("{} All nodes done".format(datetime.now().isoformat()))
        break

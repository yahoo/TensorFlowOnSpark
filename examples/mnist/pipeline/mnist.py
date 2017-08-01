# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

sess = None
def print_log(worker_num, arg):
  print("{0}: {1}".format(worker_num, arg))

def map_fun(args, iterator):
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time

  IMAGE_PIXELS=28

  # Parameters
  hidden_units = 128
  batch_size   = args.batch_size

  def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    xs = xs/255.0
    ys = numpy.array(labels)
    ys = ys.astype(numpy.uint8)
    return (xs, ys)

  global sess, x, y_, label, prediction, accuracy, hid_w, hid_b, sm_w, sm_b
  print("==== sess: {0}".format(sess))
  if sess is None:
    # Variables of the hidden layer
    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                          stddev=1.0 / IMAGE_PIXELS), name="hid_w")
    hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                          stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Placeholders or QueueRunner/Readers for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

    global_step = tf.Variable(0)

    loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    # Test trained model
    label = tf.argmax(y_, 1, name="label")
    prediction = tf.argmax(y, 1,name="prediction")
    correct_prediction = tf.equal(prediction, label)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

  results = []
  if sess == None:
    sess = tf.Session()
    ckpt = tf.train.latest_checkpoint(args.model)
    print("==== {0} latest ckpt: {1}".format(args.model, ckpt))
    if ckpt is not None:
      print("==== restoring model from {0}".format(ckpt))
      saver = tf.train.Saver([hid_w, hid_b, sm_w, sm_b])
      saver.restore(sess, ckpt)
      print("==== session ready")

  batch = []
  for item in iterator:
    batch.append(item)
    if len(batch) >= batch_size:
      print("==== processing batch")
      batch_xs, batch_ys = feed_dict(batch)
      feed = {x: batch_xs, y_: batch_ys}
      labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)
      results.extend(["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)])
      batch = []

  if len(batch) >= 0:
    print("==== processing last batch")
    batch_xs, batch_ys = feed_dict(batch)
    feed = {x: batch_xs, y_: batch_ys}
    labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)
    results.extend(["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)])

  print(results)
  print("==== returning results")
  return results

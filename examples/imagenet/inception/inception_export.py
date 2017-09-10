# Based on inception_eval.py
"""A library to export an Inception saved_model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import sys
import time


import numpy as np
import tensorflow as tf
from tensorflowonspark import TFNode

from inception import image_processing
from inception import inception_model as inception
from inception.imagenet_data import ImagenetData


tf.app.flags.DEFINE_string('export_dir', '/tmp/imagenet_export',
                           """Directory where to write saved_model.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

def export(args, argv):
  sys.argv = argv
  FLAGS = tf.app.flags.FLAGS

  """Evaluate model on Dataset for a number of steps."""
  #with tf.Graph().as_default():
  tf.reset_default_graph()

  # Get images and labels from the dataset.
  height = FLAGS.image_size
  width = FLAGS.image_size
  depth = 3

  flattened_images = tf.placeholder(tf.float32, [None, height * width * depth])
  images = tf.reshape(flattened_images, [-1, height, width, depth])
  labels = tf.placeholder(tf.int32, [None])

  # Number of classes in the Dataset label set plus 1.
  # Label 0 is reserved for an (unused) background class.
  dataset = ImagenetData(subset=FLAGS.subset)

  num_classes = dataset.num_classes() + 1

  # Build a Graph that computes the logits predictions from the
  # inference model.
  logits, _ = inception.inference(images, num_classes)

  # Calculate predictions.
  top_1_op = tf.nn.in_top_k(logits, labels, 1)
  top_5_op = tf.nn.in_top_k(logits, labels, 5)

  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(
      inception.MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()

  #graph_def = tf.get_default_graph().as_graph_def()

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if not ckpt or not ckpt.model_checkpoint_path:
      raise Exception("No checkpoint file found at: {}".format(FLAGS.train_dir))
    print("ckpt.model_checkpoint_path: {0}".format(ckpt.model_checkpoint_path))

    saver.restore(sess, ckpt.model_checkpoint_path)

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Successfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, global_step))

    print("Exporting saved_model to: {}".format(args.export_dir))
    # exported signatures defined in code
    signatures = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
        'inputs': { 'flattened_images': flattened_images },
        'outputs': { 'logits': logits },
        'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      }
    }
    TFNode.export_saved_model(sess,
                              args.export_dir,
                              tf.saved_model.tag_constants.SERVING,
                              signatures)
    print("Exported saved_model")

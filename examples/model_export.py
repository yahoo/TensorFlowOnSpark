# Copyright 2018 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import argparse
import json
import sys
import tensorflow as tf
from tensorflowonspark import TFNode

#
# Utility to load a TensorFlow checkpoint and export it as a saved_model,
# given a user-supplied signature definition in JSON format supplied as
# a command-line argument or as a file.
#
def main(_):
  # restore graph/session from checkpoint
  sess = tf.Session(graph=tf.get_default_graph())
  ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
  saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
  saver.restore(sess, ckpt)
  g = sess.graph

  # if --show, dump out all operations in this graph
  if FLAGS.show:
    for o in g.get_operations():
      print("{:>64}\t{}".format(o.name, o.type))

  if FLAGS.export_dir and FLAGS.signatures:
    # load/parse JSON signatures
    if ':' in FLAGS.signatures:
      # assume JSON string, since unix filenames shouldn't contain colons
      signatures = json.loads(FLAGS.signatures)
    else:
      # assume JSON file
      with open(FLAGS.signatures) as f:
        signatures = json.load(f)

    # convert string input/output values with actual tensors from graph
    for name, sig in signatures.items():
      for k, v in sig['inputs'].items():
        tensor_name = v if v.endswith(':0') else v + ':0'
        sig['inputs'][k] = g.get_tensor_by_name(tensor_name)
      for k, v in sig['outputs'].items():
        tensor_name = v if v.endswith(':0') else v + ':0'
        sig['outputs'][k] = g.get_tensor_by_name(tensor_name)

    # export a saved model
    TFNode.export_saved_model(sess,
                              FLAGS.export_dir,
                              tf.saved_model.tag_constants.SERVING,
                              signatures)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', type=str, help='Path to trained model checkpoint', required=True)
  parser.add_argument('--export_dir', type=str, help='Path to export saved_model')
  parser.add_argument('--signatures', type=str, help='JSON file or string representing list of signatures (inputs, outputs) to export')
  parser.add_argument('--show', help='Print all graph operations', action="store_true")
  FLAGS, _ = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv)


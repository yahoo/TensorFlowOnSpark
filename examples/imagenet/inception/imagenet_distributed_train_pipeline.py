# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from tensorflowonspark import TFCluster, TFNode, dfutil
from tensorflowonspark.pipeline import TFEstimator
from datetime import datetime

import inception_export

import os
import sys
import tensorflow as tf
import time

def main_fun(argv, ctx):

  # extract node metadata from ctx
  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  assert job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  from inception import inception_distributed_train
  from inception.imagenet_data import ImagenetData
  import tensorflow as tf

  # instantiate FLAGS on workers using argv from driver and add job_name and task_id
  print("argv:", argv)
  sys.argv = argv

  FLAGS = tf.app.flags.FLAGS
  FLAGS.job_name = job_name
  FLAGS.task_id = task_index
  print("FLAGS:", FLAGS.__dict__['__flags'])

  # Get TF cluster and server instances
  cluster_spec, server = TFNode.start_cluster_server(ctx, FLAGS.num_gpus, FLAGS.rdma)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_distributed_train.train(server.target, dataset, cluster_spec, ctx)


def preprocess(argv, ctx):
    import tensorflow as tf
    from tensorflowonspark import TFNode

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx)

    def preprocess_image(image_buffer):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image_size = 299
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
        image = tf.squeeze(image, [0])
        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        flattened = tf.reshape(image, [image_size * image_size * 3])
        return flattened

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    with tf.Session() as sess:
        tf_feed = TFNode.DataFeed(ctx.mgr, False)
        while not tf_feed.should_stop():
            examples = tf_feed.next_batch(1)
            feed_dict = { serialized_tf_example: examples }
            imgs = sess.run(images, feed_dict=feed_dict)
            tf_feed.batch_results([imgs.tolist()])    # convert to python type


if __name__ == '__main__':
  # parse arguments needed by the Spark driver
  import argparse

  sc = SparkContext(conf=SparkConf().setAppName('imagenet_distributed_train'))
  spark = SparkSession.builder.getOrCreate()
  num_executors = int(sc._conf.get("spark.executor.instances"))

  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--num_ps", help="number of PS nodes in cluster", type=int, default=1)
  parser.add_argument("--input_mode", help="method to ingest data: (spark|tf)", choices=["spark","tf"], default="tf")
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--train_dir", help="HDFS path to save/load model during train/inference", type=str)
  parser.add_argument("--export_dir", help="HDFS path to export model", type=str)
  parser.add_argument("--tfrecord_dir", help="HDFS path to temporarily save DataFrame to disk", type=str)
  parser.add_argument("--validation_data", help="HDFS path to validation data", type=str)
  parser.add_argument("--output", help="HDFS path to save output predictions", type=str)

  (args,rem) = parser.parse_known_args()

  input_mode = TFCluster.InputMode.SPARK if args.input_mode == 'spark' else TFCluster.InputMode.TENSORFLOW

  print("{0} ===== Start".format(datetime.now().isoformat()))

  df = sc.parallelize([("empty",)],1).toDF(["empty"])
  estimator = TFEstimator(main_fun, args, tf_argv=sys.argv, export_fn=inception_export.export) \
          .setModelDir(args.train_dir) \
          .setExportDir(args.export_dir) \
          .setTFRecordDir(args.tfrecord_dir) \
          .setClusterSize(args.cluster_size) \
          .setNumPS(args.num_ps) \
          .setInputMode(TFCluster.InputMode.TENSORFLOW) \
          .setTensorboard(args.tensorboard) \

  print("{0} ===== Train".format(datetime.now().isoformat()))
  model = estimator.fit(df)

  print("{0} ===== Preparing data".format(datetime.now().isoformat()))
  # df = dfutil.loadTFRecords(sc, args.validation_data)         # `TypeError: Can not infer schema for type: <class 'bytes'>`
  tfr_rdd = sc.newAPIHadoopFile(args.validation_data, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                          keyClass="org.apache.hadoop.io.BytesWritable",
                          valueClass="org.apache.hadoop.io.NullWritable").map(lambda row: bytes(row[0]))
  cluster = TFCluster.run(sc, preprocess, sys.argv, num_executors, 0, args.tensorboard, TFCluster.InputMode.SPARK)
  image_rdd = cluster.inference(tfr_rdd).persist()
#  image_rdd = cluster.inference(tfr_rdd)

#  print(">>>> image: {}".format(image_rdd.take(1)[0]))
#  print(">>>> count: {}".format(image_rdd.count()))

  schema = StructType().add("images", ArrayType(FloatType()))
  image_df = spark.createDataFrame(image_rdd, schema).persist()
#  print("image_df.schema: {}".format(image_df.schema))
#  print("image_df.image: {}".format(image_df.take(1)[0]))
#  print("image_df.count: {}".format(image_df.count()))

  print("{0} ===== Saving Dataframe".format(datetime.now().isoformat()))
#  image_df.write.json("imagenet_df")

  cluster.shutdown()

  print("{0} ===== Inference".format(datetime.now().isoformat()))
  preds = model.setTagSet(tf.saved_model.tag_constants.SERVING) \
              .setSignatureDefKey(tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY) \
              .setInputMapping({'images': 'flattened_images'}) \
              .setOutputMapping({'logits': 'output'}) \
              .transform(image_df)
  preds.write.json(args.output)


  print("{0} ===== Stop".format(datetime.now().isoformat()))

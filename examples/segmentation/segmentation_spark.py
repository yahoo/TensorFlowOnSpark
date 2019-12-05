# Copyright 2019 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  from tensorflow_examples.models.pix2pix import pix2pix
  import json
  import os
  import tensorflow_datasets as tfds
  import tensorflow as tf
  import time

  print("TensorFlow version: ", tf.__version__)
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

  def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/128.0 - 1
    input_mask -= 1
    return input_image, input_mask

  @tf.function
  def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_left_right(input_image)
      input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

  def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

  TRAIN_LENGTH = info.splits['train'].num_examples
  BATCH_SIZE = args.batch_size
  BUFFER_SIZE = args.buffer_size
  STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

  train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test = dataset['test'].map(load_image_test)

  train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_dataset = test.batch(BATCH_SIZE)

  OUTPUT_CHANNELS = 3

  with strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    def unet_model(output_channels):

      # This is the last layer of the model
      last = tf.keras.layers.Conv2DTranspose(
          output_channels, 3, strides=2,
          padding='same', activation='softmax')  # 64x64 -> 128x128

      inputs = tf.keras.layers.Input(shape=[128, 128, 3])
      x = inputs

      # Downsampling through the model
      skips = down_stack(x)
      x = skips[-1]
      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)

    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Training only (since we're using command-line)
# def create_mask(pred_mask):
#   pred_mask = tf.argmax(pred_mask, axis=-1)
#   pred_mask = pred_mask[..., tf.newaxis]
#   return pred_mask[0]
#
#
# def show_predictions(dataset=None, num=1):
#   if dataset:
#     for image, mask in dataset.take(num):
#       pred_mask = model.predict(image)
#       display([image[0], mask[0], create_mask(pred_mask)])
#   else:
#     display([sample_image, sample_mask,
#              create_mask(model.predict(sample_image[tf.newaxis, ...]))])
#
#
# class DisplayCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     clear_output(wait=True)
#     show_predictions()
#     print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
#

  EPOCHS = args.epochs
  VAL_SUBSPLITS = 5
  VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

  tf.io.gfile.makedirs(args.model_dir)
  filepath = args.model_dir + "/weights-{epoch:04d}"
  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)

  model_history = model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            callbacks=[ckpt_callback],
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset)

  if tf.__version__ == '2.0.0':
    # Workaround for: https://github.com/tensorflow/tensorflow/issues/30251
    # Save model locally as h5py and reload it w/o distribution strategy
    if ctx.job_name == 'chief':
      model.save(args.model_dir + ".h5")
      new_model = tf.keras.models.load_model(args.model_dir + ".h5")
      tf.keras.experimental.export_saved_model(new_model, args.export_dir)
  else:
    model.save(args.export_dir, save_format='tf')


if __name__ == '__main__':
  import argparse
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFCluster

  sc = SparkContext(conf=SparkConf().setAppName("segmentation"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=1000)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
  parser.add_argument("--model_dir", help="path to save model/checkpoint", default="segmentation_model")
  parser.add_argument("--export_dir", help="path to export saved_model", default="segmentation_export")
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  args = parser.parse_args()
  print("args:", args)

  cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps=0, tensorboard=args.tensorboard, input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
  cluster.shutdown(grace_secs=30)

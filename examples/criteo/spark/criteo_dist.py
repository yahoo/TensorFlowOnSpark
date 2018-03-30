# Copyright 2018 Criteo
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
# Distributed Criteo Display CTR prediction on grid based on TensorFlow on Spark
# https://github.com/yahoo/TensorFlowOnSpark

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

validation_file = None


def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))


def map_fun(args, ctx):
    from datetime import datetime
    import math
    import tensorflow as tf
    import numpy as np
    import time
    from sklearn.metrics import roc_auc_score
    import mmh3

    class CircularFile(object):
        def __init__(self, filename):
            self.filename = filename
            self.file = None

        def readline(self):
            if (self.file is None):
                self.file = tf.gfile.GFile(self.filename, "r")

            p_line = self.file.readline()

            if p_line == "":
                self.file.close()
                self.file = tf.gfile.GFile(self.filename, "r")
                p_line = self.file.readline()
            return p_line

        def close(self):
            self.file.close()
            self.file = None


    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index


    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    vocabulary_size = 39
    # Feature indexes as defined in input file
    INDEX_CAT_FEATURES = 13

    # These parameters values have been selected for illustration purpose and have not been tuned.
    learning_rate = 0.0005
    droupout_rate = 0.4
    NB_OF_HASHES_CAT = 2 ** 15
    NB_OF_HASHES_CROSS = 2 ** 15
    NB_BUCKETS = 40

    boundaries_bucket = [1.5 ** j - 0.51 for j in range(NB_BUCKETS)]
    # Same as in:
    # [https://github.com/GoogleCloudPlatform/cloudml-samples/blob/c272e9f3bf670404fb1570698d8808ab62f0fc9a/criteo_tft/trainer/task.py#L163]

    nb_input_features = ((INDEX_CAT_FEATURES) * NB_BUCKETS) + (
            (vocabulary_size - INDEX_CAT_FEATURES) * NB_OF_HASHES_CAT) + NB_OF_HASHES_CROSS


    batch_size = args.batch_size

    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)


    def get_index_bucket(feature_value):
        """
        maps the input feature to a one hot encoding index
        :param feature_value: the value of the feature
        :return: the index of the one hot encoding that activates for the input value
        """
        for index, boundary_value in enumerate(boundaries_bucket):
            if feature_value < boundary_value:
                return index
        return index


    def get_batch_validation(batch_size):
        """
        :param batch_size:
        :return: a list of read lines, each lines being a list of the features as read from the input file
        """
        global validation_file
        if validation_file is None:
            validation_file = CircularFile(args.validation)
        return [validation_file.readline().split('\t') for _ in range(batch_size)]

    def get_cross_feature_name(index, features):
        if index < INDEX_CAT_FEATURES:
            index_str = str(index) + "_" + str(get_index_bucket(int(features[index])))
        else:
            index_str = str(index) + "_" + features[index]

        return index_str

    def get_next_batch(batch):
        """
        maps the batch read from the input file to a data array, and a label array that are fed to
        the tf placeholders
        :param batch:
        :return:
        """
        data = np.zeros((batch_size, nb_input_features))
        labels = np.zeros(batch_size)

        index = 0
        while True:

            features = batch[index][1:]

            if len(features) != vocabulary_size:
                continue

            # BUCKETIZE CONTINIOUS FEATURES
            for f_index in range(0, INDEX_CAT_FEATURES ):
                if features[f_index]:
                    bucket_index = get_index_bucket(int(features[f_index]))
                    bucket_number_index = f_index * NB_BUCKETS
                    bucket_index_offset = bucket_index + bucket_number_index
                    data[index, bucket_index_offset] = 1

            # BUCKETIZE CATEGORY FEATURES
            offset = INDEX_CAT_FEATURES * NB_BUCKETS
            for f_index in range(INDEX_CAT_FEATURES, vocabulary_size):
                if features[f_index]:
                    hash_index = mmh3.hash(features[f_index]) % NB_OF_HASHES_CAT
                    hash_number_index = (f_index - INDEX_CAT_FEATURES) * NB_OF_HASHES_CAT + offset
                    hash_index_offset = hash_index + hash_number_index
                    data[index, hash_index_offset] = 1

            # BUCKETIZE CROSS CATEGORY AND CONTINIOUS
            offset = INDEX_CAT_FEATURES * NB_BUCKETS + (vocabulary_size - INDEX_CAT_FEATURES) * NB_OF_HASHES_CAT

            for index_i in range(0, vocabulary_size-1):
                for index_j in range(index_i + 1, vocabulary_size):
                    if features[index_i].rstrip() == '' or features[index_j].rstrip() == '':
                        continue

                    index_str_i = get_cross_feature_name(index_i,features)
                    index_str_j = get_cross_feature_name(index_j,features)

                    hash_index = mmh3.hash(index_str_i + "_" + index_str_j) % NB_OF_HASHES_CROSS + offset
                    data[index, hash_index] = 1

            labels[index] = batch[index][0]
            index += 1
            if index == batch_size:
                break

        return data.astype(int), labels.astype(int)



    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        is_chiefing = (task_index == 0)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            def lineartf(x, droupout_rate, is_training, name=None, reuse=None, dropout=None):
                """
                    Apply a simple lineartf transformation A*x+b to the input
                """
                n_output = 1
                if len(x.get_shape()) != 2:
                    x = tf.contrib.layers.flatten(x)

                n_input = x.get_shape().as_list()[1]

                with tf.variable_scope(name, reuse=reuse):
                    W = tf.get_variable(
                        name='W',
                        shape=[n_input, n_output],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

                    b = tf.get_variable(
                        name='b',
                        shape=[n_output],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))

                    h = tf.nn.bias_add(
                        name='h',
                        value=tf.matmul(x, W),
                        bias=b)

                    if dropout:
                        h = tf.cond(is_training, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                                    lambda: tf.layers.dropout(h, rate=0.0, training=True))

                return h, W

            is_training = tf.placeholder(tf.bool, shape=())
            input_features = tf.placeholder(tf.float32, [None, nb_input_features], name="input_features")
            input_features_lineartf, _ = lineartf(input_features, droupout_rate=droupout_rate,
                                                  name='linear_layer',
                                                  is_training=is_training,
                                                  dropout=None)

            y_true = tf.placeholder(tf.float32, shape=None)
            y_prediction = input_features_lineartf
            pCTR = tf.nn.sigmoid(y_prediction, name="pCTR")
            global_step = tf.Variable(0)
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction))
            tf.summary.scalar('cross_entropy', cross_entropy)
            adam_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,
                                                                                    global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        logdir = ctx.absolute_path(args.model)
        print("Tensorflow model path: {0}".format(logdir))

        if job_name == "worker" and is_chiefing:
            summary_writer = tf.summary.FileWriter(logdir + "/train", graph=tf.get_default_graph())
            summary_val_writer = tf.summary.FileWriter(logdir + "/validation", graph=tf.get_default_graph())

        options = dict(is_chief=is_chiefing,
                       logdir=logdir,
                       summary_op=None,
                       saver=saver,
                       global_step=global_step,
                       stop_grace_secs=300,
                       save_model_secs=0)

        if args.mode == "train":
            options['save_model_secs'] = 120
            options['init_op'] = init_op
            options['summary_writer'] = None

        sv = tf.train.Supervisor(**options)

        with sv.managed_session(server.target) as sess:

            print("{0} session ready".format(datetime.now().isoformat()))

            tf_feed = ctx.get_data_feed(args.mode == "train")
            step = 0
            while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
                batch_data, batch_labels = get_next_batch(tf_feed.next_batch(batch_size))

                if len(batch_data) > 0:

                    if args.mode == "train":

                        if sv.is_chief:
                            # Evaluate current state of the model on next batch of validation
                            batch_val = get_batch_validation(batch_size)
                            batch_data, batch_labels = get_next_batch(batch_val)
                            feed = {input_features: batch_data, y_true: batch_labels, is_training: False}
                            logloss, summary, step = sess.run([cross_entropy, summary_op, global_step], feed_dict=feed)
                            summary_val_writer.add_summary(summary, step)
                            print("validation loss: {0}".format(logloss))

                        feed = {input_features: batch_data, y_true: batch_labels, is_training: True}
                        _, logloss, summary, step = sess.run([adam_train_step, cross_entropy, summary_op, global_step],
                                                             feed_dict=feed)

                    else:
                        feed = {input_features: batch_data, y_true: batch_labels, is_training: False}
                        yscore = sess.run(pCTR, feed_dict=feed)
                        tf_feed.batch_results(yscore)

            if sv.should_stop() or step >= args.steps:
                tf_feed.terminate()
                if is_chiefing:
                    summary_writer.close()
                    summary_val_writer.close()

        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()

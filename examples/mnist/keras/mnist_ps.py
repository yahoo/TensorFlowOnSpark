# From: https://www.tensorflow.org/tutorials/distribute/parameter_server_training

def map_fun(ctx, args):
    import os
    import random
    import tensorflow as tf
    from tensorflow.keras.layers.experimental import preprocessing

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    if cluster_resolver.task_type in ("worker", "ps"):
        # Start a TensorFlow server and wait.
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True)
        server.join()
    elif cluster_resolver.task_type == "evaluator":
        print("Run side-car evaluation")
    else:
        strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

        def dataset_fn(input_context):
          global_batch_size = 64
          batch_size = input_context.get_per_replica_batch_size(global_batch_size)
          x = tf.random.uniform((10, 10))
          y = tf.random.uniform((10,))
          dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
          dataset = dataset.shard(
              input_context.num_input_pipelines,
              input_context.input_pipeline_id)
          dataset = dataset.batch(batch_size)
          dataset = dataset.prefetch(2)
          return dataset

        dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

        with strategy.scope():
            model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

        model.compile(tf.keras.optimizers.SGD(), loss='mse', steps_per_execution=10)

        working_dir = '/tmp/my_working_dir'
        log_dir = os.path.join(working_dir, 'log')
        ckpt_filepath = os.path.join(working_dir, 'ckpt')
        # backup_dir = os.path.join(working_dir, 'backup')

        # Not currently working:
        #    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)
        ]

        model.fit(dc, epochs=5, steps_per_epoch=20, callbacks=callbacks)

        # Custom Training Loop

        feature_vocab = [
            "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong", "wonder_woman"
        ]
        label_vocab = ["yes", "no"]

        with strategy.scope():
          feature_lookup_layer = preprocessing.StringLookup(
              vocabulary=feature_vocab,
              mask_token=None)
          label_lookup_layer = preprocessing.StringLookup(
              vocabulary=label_vocab,
              num_oov_indices=0,
              mask_token=None)

          raw_feature_input = tf.keras.layers.Input(
              shape=(3,),
              dtype=tf.string,
              name="feature")
          feature_id_input = feature_lookup_layer(raw_feature_input)
          feature_preprocess_stage = tf.keras.Model(
              {"features": raw_feature_input},
              feature_id_input)

          raw_label_input = tf.keras.layers.Input(
              shape=(1,),
              dtype=tf.string,
              name="label")
          label_id_input = label_lookup_layer(raw_label_input)

          label_preprocess_stage = tf.keras.Model(
              {"label": raw_label_input},
              label_id_input)

        def feature_and_label_gen(num_examples=200):
          examples = {"features": [], "label": []}
          for _ in range(num_examples):
            features = random.sample(feature_vocab, 3)
            label = ["yes"] if "avenger" in features else ["no"]
            examples["features"].append(features)
            examples["label"].append(label)
          return examples

        examples = feature_and_label_gen()

        def dataset_fn(_):
          raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

          train_dataset = raw_dataset.map(
              lambda x: (
                  {"features": feature_preprocess_stage(x["features"])},
                  label_preprocess_stage(x["label"])
              )).shuffle(200).batch(32).repeat()
          return train_dataset

        # These variables created under the `strategy.scope` will be placed on parameter
        # servers in a round-robin fashion.
        with strategy.scope():
          # Create the model. The input needs to be compatible with Keras processing layers.
          model_input = tf.keras.layers.Input(
              shape=(3,), dtype=tf.int64, name="model_input")

          emb_layer = tf.keras.layers.Embedding(
              input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
          emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
          dense_output = tf.keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
          model = tf.keras.Model({"features": model_input}, dense_output)

          optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
          accuracy = tf.keras.metrics.Accuracy()

        model.summary()

        print("emb_layer.weights: {}".format(emb_layer.weights))
    #    assert len(emb_layer.weights) == 2
    #    assert emb_layer.weights[0].shape == (4, 16384)
    #    assert emb_layer.weights[1].shape == (4, 16384)
    #    assert emb_layer.weights[0].device == "/job:ps/replica:0/task:0/device:CPU:0"
    #    assert emb_layer.weights[1].device == "/job:ps/replica:0/task:1/device:CPU:0"

        @tf.function
        def step_fn(iterator):

          def replica_fn(batch_data, labels):
            with tf.GradientTape() as tape:
              pred = model(batch_data, training=True)
              per_example_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
              loss = tf.nn.compute_average_loss(per_example_loss)
              gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
            accuracy.update_state(labels, actual_pred)
            return loss

          batch_data, labels = next(iterator)
          losses = strategy.run(replica_fn, args=(batch_data, labels))
          return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

        coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

        @tf.function
        def per_worker_dataset_fn():
          return strategy.distribute_datasets_from_function(dataset_fn)

        per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
        per_worker_iterator = iter(per_worker_dataset)

        num_epoches = 4
        steps_per_epoch = 500
        for i in range(num_epoches):
          accuracy.reset_states()
          print("Running for {} steps_per_epoch".format(steps_per_epoch))
          for _ in range(steps_per_epoch):
            coordinator.schedule(step_fn, args=(per_worker_iterator,))
          # Wait at epoch boundaries.
          coordinator.join()
          print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

        loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
        print("Final loss is %f" % loss.fetch())


if __name__ == "__main__":
    import os
    import argparse
    from pyspark.sql import SparkSession
    from tensorflowonspark import TFCluster

    # Setup parser for arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to save/load model", default="_model")

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    num_executors = 3

    # Parse args for training
    args = parser.parse_args(['--model', 'test_model'])

    # Train using a TensorFlow cluster w/ a single parameter server
    cluster = TFCluster.run(sc, map_fun, args,
                            num_executors=num_executors,
                            num_ps=1,
                            tensorboard=False,
                            input_mode=TFCluster.InputMode.TENSORFLOW,
                            log_dir="/tmp/{}/{}".format(os.environ['USER'], args.model),
                            master_node='chief',
                            persistent_nodes=['ps', 'worker', 'evaluator'])

    # The cluster will only be shutdown when the training is actually completed
    # It will take a few minutes
    cluster.shutdown()

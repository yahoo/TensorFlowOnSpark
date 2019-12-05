# Image Segmentation

Original Source: https://www.tensorflow.org/tutorials/images/segmentation

This code is based on the [Image Segmentation](https://www.tensorflow.org/tutorials/images/segmentation) notebook example, converted to a single-node TensorFlow python app, then converted into a distributed TensorFlow app using the `MultiWorkerMirroredStrategy`, and then finally adapted for TensorFlowOnSpark.  Compare the different versions to see the conversion steps involved at each stage.

Notes: 
- this example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.

#### Train via Single-Node

The [segmentation.py](segmentation.py) file contains the bulk of the code from the example notebook, minus any code for interactively visualizing the images and masks, since the end goal will be a non-interactive job in Spark.

Run the single-node example to ensure that your environment is set up correctly.  For brevity, this example only trains a single epoch (vs. the original 20 epochs), but you can modify the source to run more epochs, if desired.
```
# train
python ${TFoS_HOME}/examples/segmentation/segmentation.py
```

This will save the model weights as `keras_weights.*` files, which you can re-use in the original notebook as follows:
```
# create a new empty model
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
show_predictions()

# load the weights
model.load_weights("/path/to/keras_weights")
show_predictions()
```

#### Train via Distributed TensorFlow

Next, the [segmentation_dist.py](segmentation_dist.py) file adds a `MultiWorkerMirroredStrategy` to enable distributed training.  For simplicity, we can simulate two different machines by using separate shell windows.  If you have multiple nodes available, you can run these commands on the separate machines (using the cluster host names instead of `localhost`).
```
# on one node/shell
export TF_CONFIG='{"cluster": { "worker": ["localhost:2222", "localhost:2223"]}, "task": {"type": "worker", "index": 0}}'
python ${TFoS_HOME}/examples/segmentation/segmentation_dist.py

# on another node/shell
export TF_CONFIG='{"cluster": { "worker": ["localhost:2222", "localhost:2223"]}, "task": {"type": "worker", "index": 1}}'
python ${TFoS_HOME}/examples/segmentation/segmentation_dist.py
```

Note that training will not start until all nodes are running and connected to the cluster.  Also note that the `MultiWorkerMirroredStrategy` is a synchronous training strategy, so each node will train a batch of data and update the model weights in lock-step with each of the other nodes.  This has implications that are beyond the scope of this tutorial.  For more information, you can read the [TensorFlow distributed training documentation](https://www.tensorflow.org/beta/tutorials/distribute/keras).  Notably, you should shard the data across the workers and adjust the per-worker batch_size to account for additional nodes in the cluster.  However, in order to minimize code changes here, this is left as an exercise for the reader.

#### Train via TensorFlowOnSpark

Next, we convert the `segmentation_dist.py` file to TensorFlowOnSpark, resulting in the [segmentation_spark.py](segmentation_spark.py) file.  Then, run in a local Spark standalone cluster as follows:
```
# Start a local standalone Spark cluster
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export TFoS_HOME=<path/to/TensorFlowOnSpark>

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

# remove any old artifacts
rm -Rf ${TFoS_HOME}/segmentation_model.h5 ${TFoS_HOME}/segmentation_model ${TFoS_HOME}/segmentation_export

# train
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
${TFoS_HOME}/examples/segmentation/segmentation_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--model_dir ${TFoS_HOME}/segmentation_model \
--export_dir ${TFoS_HOME}/segmentation_export \
--epochs 1

# Shutdown the Spark Standalone cluster
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

Once again, this only trains a single epoch and doesn't adjust for the increased cluster size.  Feel free to experiment on your own.

This example will save the model in several different formats:
- TensorFlow/Keras checkpoint (`segmentation_model`)
- Keras HDF5 file (`segmentation_model.h5`)
- TensorFlow saved_model (`segmentation_export`)

You can re-load these into the original notebook example (for visualization of the segmentation masks) with the following code:
```
# segmentation_model
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights("/path/to/segmentation_model/weights-0001")
show_predictions(test_dataset)

# segmentation_model.h5
model = tf.keras.models.load_model("/path/to/segmentation_model.h5")
show_predictions(test_dataset)

# segmentation_export
model = tf.keras.experimental.load_from_saved_model("/path/to/segmentation_export")
show_predictions(test_dataset)
```

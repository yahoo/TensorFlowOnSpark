# ResNet Image Classification

Original Source: https://github.com/tensorflow/models/tree/master/official/vision/image_classification

This code is based on the Image Classification model from the official [TensorFlow Models](https://github.com/tensorflow/models) repository.  This example already supports different forms of distribution via the `DistributionStrategy` API, so there isn't much additional work to convert it to TensorFlowOnSpark.

Notes: 
- This example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.
- For simplicity, this just uses a single-node Spark Standalone installation.

#### Run the Single-Node Application

First, make sure that you can run the example per the [original instructions](https://github.com/tensorflow/models/tree/68c3c65596b8fc624be15aef6eac3dc8952cbf23/official/vision/image_classification).  For now, we'll just use the CIFAR-10 dataset.  After cloning the `tensorflow/models` repository and downloading the dataset, you should be able to run the training as follows:
```
export TENSORFLOW_MODELS=/path/to/tensorflow/models
export CIFAR_DATA=/path/to/cifar
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
python resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --train_epochs=1
```

If you have GPUs available, just set `--num_gpus` to the number of GPUs on your machine.  Note: by default, `--train_epochs=182`, which runs for a long time on a CPU machine, so for brevity, we'll just run a single epoch in these examples.

#### Run as a Distributed TensorFlow Application

Next, confirm that this application is capable of being distributed.  We can test this on a single CPU machine by using two different terminal/shell sessions, as follows:
```
# in one shell/window
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
export TF_CONFIG='{"cluster": { "worker": ["localhost:2222", "localhost:2223"]}, "task": {"type": "worker", "index": 0}}'
python resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --ds=multi_worker_mirrored --train_epochs=1

# in another shell/window
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
export TF_CONFIG='{"cluster": { "worker": ["localhost:2222", "localhost:2223"]}, "task": {"type": "worker", "index": 1}}'
python resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --ds=multi_worker_mirrored --train_epochs=1
```

Note that we now configure the code to use the `MultiWorkerMirroredtrategy`.  Also note that training will not begin until both nodes have started.

### Run as a TensorFlowOnSpark Application

Finally, we can run the converted application as follows:
```
export TFoS_HOME=/path/to/TensorFlowOnSpark
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))

# start spark cluster
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

# train and evaluate
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--py-files ${TFoS_HOME}/examples/resnet/resnet_cifar_dist.py \
${TFoS_HOME}/examples/resnet/resnet_cifar_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--epochs 1 \
--data_dir /Users/leewyang/datasets/cifar10/cifar-10-batches-bin \
--num_gpus=0 \
--ds=multi_worker_mirrored \
--train_epochs 1

# shutdown spark
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

Notes:
- Most of the original TensorFlow code from `resnet_cifar_main.py` has been copied into `resnet_cifar_dist.py`, so you can diff the changes.
- The `def main(_)` function was changed to `def main_fun(argv, ctx)`.
- The `absl_app.run(main)` invocation was replaced by the Spark "main" function in `resnet_cifar_spark.py`.  This file mostly contains the Spark application boilerplate along with the TensorFlowOnSpark calls to setup the TensorFlow cluster.  Note that having the separate Spark and TensorFlow files can help isolate code and avoid Spark serialization issues.
- The Spark "main" function uses `argparse` to parse TensorFlowOnSpark-specific command line arguments, but it passes the remaining argments (in the `rem` variable) to the TensorFlow `main_fun`, which then parses those arguments via `define_cifar_flags()` and `flags.FLAGS(argv)`.
- In a truly distributed environment, you would need:
  - A distributed file system to store the dataset, so that each executor/node is able to read the data.
  - The dependencies from the `tensorflow/models` to be available on the executors, either installed locally or bundled with the Spark application.

# ResNet Image Classification

Original Source: https://github.com/tensorflow/models/tree/master/official/benchmark/models

This code is based on the Image Classification model from the official [TensorFlow Models](https://github.com/tensorflow/models) repository.  This example already supports different forms of distribution via the `DistributionStrategy` API, so there isn't much additional work to convert it to TensorFlowOnSpark.

Notes:
- This example assumes that Spark, TensorFlow, and TensorFlowOnSpark are already installed.
- For simplicity, this just uses a single-node Spark Standalone installation.

#### Run the Single-Node Application

First, make sure that you can run the original example, as follows:
```
# clone the TensorFlow models repository
git clone https://github.com/tensorflow/models
cd models

# checkout the specific revision that this example was based upon
git checkout c25c3e882e398d287240f619d7f56ac5b2973b6e

# download the CIFAR10 dataset to /tmp/cifar10_data
python official/r1/resnet/cifar10_download_and_extract.py

# run the example
export TENSORFLOW_MODELS=$(pwd)
export CIFAR_DATA=/tmp/cifar10_data/cifar-10-batches-bin
export PYTHONPATH=${TENSORFLOW_MODELS}:$PYTHONPATH

# pip install tensorflow==2.1.1 tensorflow_model_optimization==0.3.0
python ${TENSORFLOW_MODELS}/official/benchmark/models/resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --train_epochs=1
```

If you have GPUs available, just set `--num_gpus` to the number of GPUs on your machine.

#### Run as a Distributed TensorFlow Application

Next, confirm that this application is capable of being distributed.  We can test this on a single CPU machine by using two different terminal/shell sessions, as follows:
```
# in one shell/window
export TFoS_HOME=/path/to/TensorFlowOnSpark
export CIFAR_DATA=/tmp/cifar10_data/cifar-10-batches-bin
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
export TF_CONFIG='{"cluster": { "chief": ["localhost:2222"], "worker": ["localhost:2223"]}, "task": {"type": "chief", "index": 0}}'
python ${TFoS_HOME}/examples/resnet/resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --ds=multi_worker_mirrored --train_epochs=1

# in another shell/window
# cd /path/to/tensorflow/models
export TFoS_HOME=/path/to/TensorFlowOnSpark
export CIFAR_DATA=/tmp/cifar10_data/cifar-10-batches-bin
export PYTHONPATH=${PYTHONPATH}:${TENSORFLOW_MODELS}
export TF_CONFIG='{"cluster": { "chief": ["localhost:2222"], "worker": ["localhost:2223"]}, "task": {"type": "worker", "index": 0}}'
python ${TFoS_HOME}/examples/resnet/resnet_cifar_main.py --data_dir=${CIFAR_DATA} --num_gpus=0 --ds=multi_worker_mirrored --train_epochs=1
```

Note that we now configure the code to use the `MultiWorkerMirroredStrategy`.  Also note that training will not begin until both nodes have started.

### Run as a TensorFlowOnSpark Application

Finally, we can run the converted application as follows:
```bash
export TFoS_HOME=/path/to/TensorFlowOnSpark
export TENSORFLOW_MODELS=/path/to/tensorflow/models
export CIFAR_DATA=/tmp/cifar10_data/cifar-10-batches-bin
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
--data_dir ${CIFAR_DATA} \
--num_gpus=0 \
--ds=multi_worker_mirrored \
--train_epochs 1

# shutdown spark
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

Notes:
- Most of the original TensorFlow code from `resnet_cifar_main.py` has been copied into `resnet_cifar_dist.py`, so you can diff the changes required for TensorFlowOnSpark.
- The `def main(_)` function was changed to `def main_fun(argv, ctx)`.
- The `absl_app.run(main)` invocation was replaced by the Spark "main" function in `resnet_cifar_spark.py`.  This file mostly contains the Spark application boilerplate along with the TensorFlowOnSpark calls to setup the TensorFlow cluster.  Note that having the separate Spark and TensorFlow files can help isolate code and avoid Spark serialization issues.
- The Spark "main" function uses `argparse` to parse TensorFlowOnSpark-specific command line arguments, but it passes the remaining argments (in the `rem` variable) to the TensorFlow `main_fun`, which then parses those arguments via `define_cifar_flags()` and `flags.FLAGS(argv)`.
- In a truly distributed environment, you would need:
  - A distributed file system to store the dataset, so that each executor/node is able to read the data.
  - The dependencies from the `tensorflow/models` to be available on the executors, either installed locally or bundled with the Spark application.

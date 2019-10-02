import resnet_cifar_dist

if __name__ == '__main__':
  # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  # absl_app.run(main)
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFCluster
  import argparse

  sc = SparkContext(conf=SparkConf().setAppName("resnet_cifar"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser()
  parser.add_argument("--cluster_size", help="number of nodes in the cluster (for Spark Standalone)", type=int, default=num_executors)
  parser.add_argument("--num_ps", help="number of parameter servers", type=int, default=1)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  args, rem = parser.parse_known_args()

  cluster = TFCluster.run(sc, resnet_cifar_dist.main_fun, rem, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)
  cluster.shutdown()

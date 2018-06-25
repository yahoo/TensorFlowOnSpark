#coding=utf-8
""" wide deep model"""


import tensorflow as tf
import numpy as np

from tensorflowonspark import TFNode
import sys,os



from official.utils.misc import model_helpers
from official.utils.logs import hooks_helper


default_data_path = "hdfs:///user/tfos/train/"
default_train_files = [default_data_path + "/part-%05d"%i for i in range(128)]


default_eval_path = "hdfs:///user/tfos/eval/"

default_eval_files = [default_eval_path + "/part-%05d"%i for i in range(128)]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

## tpindex data 53 columns
_CSV_COLUMNS = [
    "fea1","fea2","fea3","fea4","fea5",
    "fea6","fea7","fea8","fea9","fea10",
    "fea11","fea12","fea13","fea14","fea15",
    "fea16","fea17","fea18","fea19","fea20",
    "fea21","fea22","fea23","fea24","fea25",
    "fea26","fea27","fea28","fea29","fea30",
    "fea31","fea32","fea33","fea34","fea35",
    "fea36","fea37","fea38","fea39","fea40",
    "fea41","fea42","fea43","fea44","fea45",
    "fea46","fea47","fea48","fea49","fea50",
    "fea51","fea52","fea53"
]


_CSV_COLUMN_DEFAULTS = [
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0'],['0'],['0'],
    ['0'],['0'],['0']
]

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}

selected  = ["fea9", "fea15", "fea16", "fea17", "fea18",
             "fea22", "fea23", "fea24", "fea2", "fea1",
             "fea7"]

def input_fn(data_file, num_epochs, shuffle, batch_size):

    def select_features(features):
        d = {}
        for k in selected:
            d[k] = features[k]
        return d
    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim="\t")
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('fea20')
        selected_features = select_features(features)
        return selected_features, tf.equal(labels, "1")

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

## bussiness oriented
bucket_size_dict = {
    "fea9": list(np.linspace(1000, 5000000, 1000)),
    "fea16":  list(np.linspace(0, 2147483647, 1000)),
    "fea18": list(np.linspace(0, 2147483647, 500)),
    "fea22": list(np.linspace(0, 10, 5)),
    "fea23": list(np.linspace(0, 150, 150)),
    "fea15": list(np.linspace(0, 2147483647, 20000)),
    "fea2": list(np.linspace(0, 10000000, 100)),
    "fea1":list(np.linspace(0, 200, 100)),
    "fea17": list(np.linspace(0,20000, 10)),
    "fea7": list(np.linspace(0,50, 10))
}

## build columns
def build_model_columns():
    def _column_name(n):
        return n + "_column"
    column_dict = {}
    for k in selected:
        column_dict[_column_name(k)] = tf.feature_column.numeric_column(key=k)
        print "column:" + _column_name(k)

    base_columns = [column_dict[_column_name(k)] for k in selected]

    def _build_bucket_column(name):
        buckets = bucket_size_dict[name]
        return tf.feature_column.bucketized_column(column_dict[_column_name(name)], buckets)

    loc_bc = _build_bucket_column("fea9")
    adv_bc = _build_bucket_column("fea16")
    fea18_bc = _build_bucket_column("fea18")
    fea22_bc = _build_bucket_column("fea22")
    fea23_bc = _build_bucket_column("fea23")
    order_bc = _build_bucket_column("fea15")
    fea2_bc = _build_bucket_column("fea2")
    plat_bc = _build_bucket_column("fea1")

    tuwen_bc = _build_bucket_column("fea17")
    fea7_bc = _build_bucket_column("fea7")

    crossed_columns = [
        tf.feature_column.crossed_column([loc_bc, adv_bc], hash_bucket_size=20000),
        tf.feature_column.crossed_column([loc_bc, order_bc], hash_bucket_size=4000000),
        tf.feature_column.crossed_column([loc_bc, fea18_bc], hash_bucket_size=50000),
        tf.feature_column.crossed_column([loc_bc, fea22_bc], hash_bucket_size=2000),
        tf.feature_column.crossed_column([loc_bc, fea23_bc], hash_bucket_size=20000),
        tf.feature_column.crossed_column([order_bc, fea22_bc], hash_bucket_size=50000),
        tf.feature_column.crossed_column([order_bc, fea23_bc], hash_bucket_size=1000000),
        tf.feature_column.crossed_column([fea18_bc, fea22_bc], hash_bucket_size=1000),
        tf.feature_column.crossed_column([fea18_bc, fea23_bc], hash_bucket_size=200000),
        tf.feature_column.crossed_column([fea22_bc, fea23_bc], hash_bucket_size=500),
        tf.feature_column.crossed_column([fea2_bc, order_bc], hash_bucket_size=500000),
        tf.feature_column.crossed_column([plat_bc, order_bc], hash_bucket_size=200000),
        tf.feature_column.crossed_column([fea7_bc, order_bc], hash_bucket_size=200000),
        tf.feature_column.crossed_column([fea2_bc, fea22_bc], hash_bucket_size=500),
        tf.feature_column.crossed_column([fea2_bc, fea23_bc], hash_bucket_size=5000),
        tf.feature_column.crossed_column([plat_bc, fea22_bc], hash_bucket_size=100),
        tf.feature_column.crossed_column([plat_bc, fea23_bc], hash_bucket_size=1000),
        tf.feature_column.crossed_column([fea7_bc, fea22_bc], hash_bucket_size=50),
        tf.feature_column.crossed_column([fea2_bc, tuwen_bc], hash_bucket_size=500),
        tf.feature_column.crossed_column([plat_bc, tuwen_bc], hash_bucket_size=500),
        tf.feature_column.crossed_column([fea7_bc, tuwen_bc], hash_bucket_size=100),
        tf.feature_column.crossed_column([fea2_bc, fea18_bc], hash_bucket_size=5000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        column_dict[_column_name("fea23")],
        column_dict[_column_name("fea22")],
        column_dict[_column_name("fea24")],
        column_dict[_column_name("fea7")],
        column_dict[_column_name("fea1")],
        column_dict[_column_name("fea16")],
    ]

    return wide_columns, deep_columns


##business oriented
feature_dimension_dict = {
    "fea9": 500,
    "fea15": 20000,
    "fea16": 1000,
    "fea17": 100,
    "fea18": 100,
    "fea22": 5,
    "fea23": 200,
    "fea24": 2000,
    "fea2": 100,
    "fea1": 50,
    "fea7": 50,
}

embedding_dimension_dict = {
    "fea9": 8,
    "fea15": 8,
    "fea16": 8,
    "fea17": 8,
    "fea18": 8,
    "fea22": 8,
    "fea23": 8,
    "fea24": 8,
    "fea2": 8,
    "fea1": 8,
    "fea7": 8,
}

def build_model_columns_hash():
    def _column_name(k):
        return k + "_column"
    def _hash_column(k):
        return tf.feature_column.categorical_column_with_hash_bucket(key=k,
                                                       hash_bucket_size=feature_dimension_dict[k])

    column_dict = {}
    ## actually some column like fea22, tuwen could use tf.feature_column.categorical_column_with_identity
    ## since these features are sparse with few dimensions, here we use hash for all feature columns
    for k in selected:
        column_dict[k] = _hash_column(k)

    base_columns = [column_dict[k] for k in selected]
    crossed_columns = [
        tf.feature_column.crossed_column(keys=["fea9", "fea16"], hash_bucket_size=20000),
        tf.feature_column.crossed_column(keys=["fea9", "fea15"], hash_bucket_size=4000000),
        tf.feature_column.crossed_column(keys=["fea9", "fea18"], hash_bucket_size=50000),
        tf.feature_column.crossed_column(keys=["fea9", "fea22"], hash_bucket_size=2000),
        tf.feature_column.crossed_column(keys=["fea9", "fea23"], hash_bucket_size=20000),
        tf.feature_column.crossed_column(keys=["fea15", "fea22"], hash_bucket_size=50000),
        tf.feature_column.crossed_column(keys=["fea15", "fea23"], hash_bucket_size=1000000),
        tf.feature_column.crossed_column(keys=["fea18", "fea22"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(keys=["fea18", "fea23"], hash_bucket_size=200000),

        tf.feature_column.crossed_column(keys=["fea22", "fea23"], hash_bucket_size=500),
        tf.feature_column.crossed_column(keys=["fea2", "fea15"], hash_bucket_size=500000),
        tf.feature_column.crossed_column(keys=["fea1", "fea15"], hash_bucket_size=200000),
        tf.feature_column.crossed_column(keys=["fea7", "fea15"], hash_bucket_size=200000),
        tf.feature_column.crossed_column(keys=["fea2", "fea22"], hash_bucket_size=500),
        tf.feature_column.crossed_column(keys=["fea2", "fea23"], hash_bucket_size=5000),
        tf.feature_column.crossed_column(keys=["fea1", "fea22"], hash_bucket_size=100),

        tf.feature_column.crossed_column(keys=["fea1", "fea23"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(keys=["fea7", "fea22"], hash_bucket_size=50),
        tf.feature_column.crossed_column(keys=["fea2", "fea17"], hash_bucket_size=500),

        tf.feature_column.crossed_column(keys=["fea1", "fea17"], hash_bucket_size=500),
        tf.feature_column.crossed_column(keys=["fea7", "fea17"], hash_bucket_size=100),
        tf.feature_column.crossed_column(keys=["fea2", "fea18"], hash_bucket_size=5000),
    ]

    wide_columns = base_columns + crossed_columns
    deep_columns = [tf.feature_column.embedding_column(column_dict[k], embedding_dimension_dict[k]) for k in selected]
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns_hash()
    hidden_units = [100, 75, 50, 25, 2]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)

def export_model(model, model_type, export_dir):
    """Export to SavedModel format.

    Args:
      model: Estimator object
      model_type: string indicating model type. "wide", "deep" or "wide_deep"
      export_dir: directory to export the model.
    """
    wide_columns, deep_columns = build_model_columns_hash()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn)

def main_func(args, ctx):
    model = build_estimator(args.model_dir, args.model_type)
    #train_file = os.path.join(args.data_dir, 'adult.data')
    #test_file = os.path.join(args.data_dir, 'adult.test')
    train_file = [name for idx, name in enumerate(default_train_files) if idx % args.task_num == ctx.task_index]
    #train_file = one_test_file
    #train_file = default_train_files
    #test_file = default_eval_files
    test_file = [name for idx, name in enumerate(default_eval_files) if idx % args.task_num == ctx.task_index]
    #test_file = one_test_file

    print "train_files:", train_file, "==============\n test_file:", test_file

    def train_input_fn():
        return input_fn(train_file, args.train_epochs, True, args.batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, args.batch_size)


    loss_prefix = LOSS_PREFIX.get(args.model_type, '')

    print("start for with: train_epochs {0}, between {1}".format(args.train_epochs, args.epochs_between_evals))

    tensors_to_log={'average_loss': loss_prefix + 'head/truediv', 'loss': loss_prefix + 'head/weighted_loss/Sum'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50000)
    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.max_steps,hooks=[logging_hook])
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None,hooks=[logging_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(model,train_spec,eval_spec)


    if args.export_dir is not None and ctx.task_index == 0:
        print("export: " + args.export_dir)
        export_model(model, args.model_type, args.export_dir)

if __name__ == '__main__':
    import argparse
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from tensorflowonspark import TFCluster

    sc = SparkContext(conf=SparkConf().setAppName("wdm_on_tfos"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    num_ps = 1


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=1024)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--train_epochs", help="number of epochs of training data", type=int, default=1)
    parser.add_argument("--epochs_between_evals", help="number of epochs of training data", type=int, default=1)

    parser.add_argument("--export_dir", help="directory to export saved_model",default="hdfs:///user/tfos/export/")
    parser.add_argument("--images", help="HDFS path to MNIST images in parallelized CSV format")
    parser.add_argument("--input_mode", help="input mode (tf|spark)", default="tf")
    parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized CSV format")
    parser.add_argument("--model_dir", help="directory to write model checkpoints",default="hdfs:///user/tfos/data/")
    parser.add_argument("--data_dir", help="directory to write model checkpoints", default="hdfs:///user/tfos/model")
    parser.add_argument("--model_type", help="directory to write model checkpoints",default="wide_deep")
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--task_num", help="number of worker nodes", type=int, default=1)
    parser.add_argument("--max_steps", help="max number of steps to train", type=int, default=2000000)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    assert(args.num_ps + args.task_num == num_executors)

    cluster = TFCluster.run(sc, main_func, args, args.cluster_size, args.num_ps, args.tensorboard, 
                    TFCluster.InputMode.TENSORFLOW, log_dir=args.model_dir, master_node='master')
    cluster.shutdown()

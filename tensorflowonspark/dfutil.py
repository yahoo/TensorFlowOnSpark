# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

loadedDF = {}       # Stores origin paths of loaded DataFrames (df => path)
savedDF = {}        # Stores destination paths of saved DataFrames (df => path)

def saveAsTFRecords(df, output_dir):
  """Helper function to persist a Spark DataFrame as TFRecords"""
  tf_rdd = df.rdd.mapPartitions(toTFExample(df.dtypes))
  tf_rdd.saveAsNewAPIHadoopFile(output_dir, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                            keyClass="org.apache.hadoop.io.BytesWritable",
                            valueClass="org.apache.hadoop.io.NullWritable")
  savedDF[tf_rdd] = output_dir


def loadTFRecords(sc, input_dir):
  """Helper function to load TFRecords from disk into a DataFrame"""
  tfr_rdd = sc.newAPIHadoopFile(input_dir, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                              keyClass="org.apache.hadoop.io.BytesWritable",
                              valueClass="org.apache.hadoop.io.NullWritable")
  df = tfr_rdd.map(lambda x: fromTFExample(str(x[0]))).toDF()
  loadedDF[df] = input_dir
  return df


def isSavedDF(df):
  return df in savedDF


def isLoadedDF(df):
  return df in loadedDF


def toTFExample(dtypes):
  """mapPartition function to convert a Spark DataFrame into serialized `tf.train.Example` bytestring.

  Note that `tf.train.Example` is a fairly flat structure with limited datatypes, e.g. `tf.train.FloatList`,
  `tf.train.Int64List`, and `tf.train.BytesList`, so most DataFrame types will be converted into one of these types.

  Args:
    dtypes: the `DataFrame.dtypes` of the source DataFrame.

  Returns:
    A mapPartition lambda function which converts the source DataFrame into `tf.train.Example` bytestring
  """
  def _toTFExample(iter):
    import tensorflow as tf

    # supported type mappings between DataFrame.dtypes and tf.train.Feature types
    float_dtypes = ['float', 'double']
    int64_dtypes = ['boolean', 'tinyint', 'smallint', 'int', 'bigint', 'long']
    bytes_dtypes = ['string']
    float_list_dtypes = ['array<float>', 'array<double>']
    int64_list_dtypes = ['array<boolean>', 'array<tinyint>', 'array<smallint>', 'array<int>', 'array<bigint>', 'array<long>']

    def _toTFFeature(name, dtype, row):
      feature = None
      if dtype in float_dtypes:
        feature = (name, tf.train.Feature(float_list=tf.train.FloatList(value=[row[name]])))
      elif dtype in int64_dtypes:
        feature = (name, tf.train.Feature(int64_list=tf.train.Int64List(value=[row[name]])))
      elif dtype in bytes_dtypes:
        feature = (name, tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(row[name])])))
      elif dtype in float_list_dtypes:
        feature = (name, tf.train.Feature(float_list=tf.train.FloatList(value=row[name])))
      elif dtype in int64_list_dtypes:
        feature = (name, tf.train.Feature(int64_list=tf.train.Int64List(value=row[name])))
      else:
        raise Exception("Unsupported dtype: {0}".format(dtype))
      return feature

    results = []
    for row in iter:
      features = dict([_toTFFeature(name, dtype, row) for name, dtype in dtypes])
      example = tf.train.Example(features=tf.train.Features(feature=features))
      results.append((bytearray(example.SerializeToString()),None))
    return results

  return _toTFExample


def fromTFExample(bytestr):
  """map function to convert an RDD of `tf.train.Example' bytestring to an RDD of Row"""
  import tensorflow as tf
  from pyspark.sql import Row

  example = tf.train.Example()
  example.ParseFromString(bytestr)

  # convert from protobuf-like dict to DataFrame-friendly dict
  def _get_value(v):
    if v.int64_list.value:
      return list(v.int64_list.value)
    elif v.float_list.value:
      return list(v.float_list.value)
    else:
      return list(v.bytes_list.value)

  d = { k: _get_value(v) for k,v in example.features.feature.items() }
  row = Row(**d)
  return row


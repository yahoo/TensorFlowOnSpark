# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""A collection of utility functions for loading/saving TensorFlow TFRecords files as Spark DataFrames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, LongType, StringType, StructField, StructType

loadedDF = {}       # Stores origin paths of loaded DataFrames (df => path)


def isLoadedDF(df):
  """Returns True if the input DataFrame was produced by the loadTFRecords() method.

  This is primarily used by the Spark ML Pipelines APIs.

  Args:
    :df: Spark Dataframe
  """
  return df in loadedDF


def saveAsTFRecords(df, output_dir):
  """Save a Spark DataFrame as TFRecords.

  This will convert the DataFrame rows to TFRecords prior to saving.

  Args:
    :df: Spark DataFrame
    :output_dir: Path to save TFRecords
  """
  tf_rdd = df.rdd.mapPartitions(toTFExample(df.dtypes))
  tf_rdd.saveAsNewAPIHadoopFile(output_dir, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")


def loadTFRecords(sc, input_dir, binary_features=[]):
  """Load TFRecords from disk into a Spark DataFrame.

  This will attempt to automatically convert the tf.train.Example features into Spark DataFrame columns of equivalent types.

  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to
  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"
  from the caller in the ``binary_features`` argument.

  Args:
    :sc: SparkContext
    :input_dir: location of TFRecords on disk.
    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.

  Returns:
    A Spark DataFrame mirroring the tf.train.Example schema.
  """
  import tensorflow as tf

  tfr_rdd = sc.newAPIHadoopFile(input_dir, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")

  # infer Spark SQL types from tf.Example
  record = tfr_rdd.take(1)[0]
  example = tf.train.Example()
  example.ParseFromString(bytes(record[0]))
  schema = infer_schema(example, binary_features)

  # convert serialized protobuf to tf.Example to Row
  example_rdd = tfr_rdd.mapPartitions(lambda x: fromTFExample(x, binary_features))

  # create a Spark DataFrame from RDD[Row]
  df = example_rdd.toDF(schema)

  # save reference of this dataframe
  loadedDF[df] = input_dir
  return df


def toTFExample(dtypes):
  """mapPartition function to convert a Spark RDD of Row into an RDD of serialized tf.train.Example bytestring.

  Note that tf.train.Example is a fairly flat structure with limited datatypes, e.g. tf.train.FloatList,
  tf.train.Int64List, and tf.train.BytesList, so most DataFrame types will be coerced into one of these types.

  Args:
    :dtypes: the DataFrame.dtypes of the source DataFrame.

  Returns:
    A mapPartition function which converts the source DataFrame into tf.train.Example bytestrings.
  """
  def _toTFExample(iter):

    # supported type mappings between DataFrame.dtypes and tf.train.Feature types
    float_dtypes = ['float', 'double']
    int64_dtypes = ['boolean', 'tinyint', 'smallint', 'int', 'bigint', 'long']
    bytes_dtypes = ['binary', 'string']
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
      results.append((bytearray(example.SerializeToString()), None))
    return results

  return _toTFExample


def infer_schema(example, binary_features=[]):
  """Given a tf.train.Example, infer the Spark DataFrame schema (StructFields).

  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to
  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"
  from the caller in the ``binary_features`` argument.

  Args:
    :example: a tf.train.Example
    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.

  Returns:
    A DataFrame StructType schema
  """
  def _infer_sql_type(k, v):
    # special handling for binary features
    if k in binary_features:
      return BinaryType()

    if v.int64_list.value:
      result = v.int64_list.value
      sql_type = LongType()
    elif v.float_list.value:
      result = v.float_list.value
      sql_type = DoubleType()
    else:
      result = v.bytes_list.value
      sql_type = StringType()

    if len(result) > 1:             # represent multi-item tensors as Spark SQL ArrayType() of base types
      return ArrayType(sql_type)
    else:                           # represent everything else as base types (and empty tensors as StringType())
      return sql_type

  return StructType([StructField(k, _infer_sql_type(k, v), True) for k, v in sorted(example.features.feature.items())])


def fromTFExample(iter, binary_features=[]):
  """mapPartition function to convert an RDD of serialized tf.train.Example bytestring into an RDD of Row.

  Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to
  disambiguate these types for Spark DataFrames DTypes (StringType and BinaryType), so we require a "hint"
  from the caller in the ``binary_features`` argument.

  Args:
    :iter: the RDD partition iterator
    :binary_features: a list of tf.train.Example features which are expected to be binary/bytearrays.

  Returns:
    An array/iterator of DataFrame Row with features converted into columns.
  """
  # convert from protobuf-like dict to DataFrame-friendly dict
  def _get_value(k, v):
    # special handling for binary features
    if k in binary_features:
      return bytearray(v.bytes_list.value[0])

    if v.int64_list.value:
      result = v.int64_list.value
    elif v.float_list.value:
      result = v.float_list.value
    else:
      result = v.bytes_list.value

    if len(result) > 1:         # represent multi-item tensors as python lists
      return list(result)
    elif len(result) == 1:      # extract scalars from single-item tensors
      return result[0]
    else:                       # represent empty tensors as python None
      return None

  results = []
  for record in iter:
    example = tf.train.Example()
    example.ParseFromString(bytes(record[0]))       # record is (bytestr, None)
    d = {k: _get_value(k, v) for k, v in sorted(example.features.feature.items())}
    row = Row(**d)
    results.append(row)

  return results

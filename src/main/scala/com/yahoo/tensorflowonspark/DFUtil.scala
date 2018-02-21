/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.apache.hadoop.io.BytesWritable
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.tensorflow.example.Feature.KindCase
import org.tensorflow.example._
import com.google.protobuf.ByteString

import scala.collection.mutable.ListBuffer

/**
  * Helper object for loading TFRecords into DataFrames.
  */
object DFUtil {

  /**
    * This will attempt to automatically convert the tf.train.Example features into Spark DataFrame columns of equivalent types.
    *
    * Note: TensorFlow represents both strings and binary types as tf.train.BytesList, and we need to disambiguate these types
    * for Spark DataFrame DTypes (StringType and BinaryType), so we require a "hint" from the caller in the `binary_features`
    * argument.
    * @param sc SparkContext
    * @param inputDir path to TFRecords on disk.
    * @param schemaHint Spark DataFrame schema used as a hint when inferring the schema.  This can be either a partial schema
    *                   or a fully-specified schema.
    * @return a Spark DataFrame representing the TFRecords from the inputDir.
    */
  def loadTFRecords(inputDir: String, schemaHint: StructType=new StructType())(implicit sc: SparkContext): DataFrame = {
    // load as RDD[Example]
    val rawRDD = sc.newAPIHadoopFile(inputDir,
                                    classOf[org.tensorflow.hadoop.io.TFRecordFileInputFormat],
                                    classOf[org.apache.hadoop.io.BytesWritable],
                                    classOf[org.apache.hadoop.io.NullWritable])

    // infer Spark SQL types from tf.Example
    val bytesRDD = rawRDD.map { case (bytes,_) => bytes.getBytes }
    val example = Example.parseFrom(bytesRDD.take(1).head)
    val schema = inferSchema(example, schemaHint)

    val exampleRDD = rawRDD.map { case (bytes, _) => Example.parseFrom(bytes.getBytes) }

    // convert RDD[Example] to RDD[Row]
    val rdd = exampleRDD.map(fromTFExample(_, schema))
    val spark = SparkSession.builder.getOrCreate()

    // create a DataFrame using the inferred schema
    spark.createDataFrame(rdd, schema)
  }

  /**
    * Given a TensorFlow Example, infers the equivalent Spark DataFrame schema, allowing for "hints" from the caller
    * to disambiguate types, e.g. BYTES_LIST can map to either StringType and BinaryType, INT64_LIST can map to either
    * IntegerType or LongType, etc.
    *
    * @param example TensorFlow Example.
    * @param schemaHint Spark schema StructType.  Any fields in this hint, will be returned with the hinted type.  Any fields
    *                   not present in this hint will be inferred with "best guesses".
    * @return a fully-specified Spark schema StructType.
    */
  def inferSchema(example: Example, schemaHint: StructType): StructType = {
    import scala.collection.JavaConversions._

    // convert schemaHint to a map for easier lookup
    val ftypes = schemaHint.map(f => (f.name, f.dataType)).toMap

    def _inferSchemaType(name: String, f: Feature): DataType = {
      // if user-hint provided, just return it
      if (ftypes.contains(name)) return ftypes(name)

      // else infer as best we can
      val dtype = f.getKindCase match {
        case KindCase.BYTES_LIST => StringType    // prefer StringType, since it's most likely?
        case KindCase.INT64_LIST => LongType
        case KindCase.FLOAT_LIST => FloatType
        case KindCase.KIND_NOT_SET => throw new Exception(s"Unsupported TFRecord type: ${f.getKindCase}")
      }

      // check if there are multiple values, i.e. array
      val isArray = dtype match {
        case BinaryType => f.getBytesList.getValueList.size > 1
        case StringType => f.getBytesList.getValueList.size > 1
        case LongType => f.getInt64List.getValueList.size > 1
        case FloatType => f.getFloatList.getValueList.size > 1
      }

      if (isArray) ArrayType(dtype) else dtype
    }

    val fmap = example.getFeatures.getFeatureMap.toMap

    // First, try to preserve order of hinted fields (e.g. if a full schema was supplied)
    val hintedFields = schemaHint.map(field =>
      StructField(field.name, _inferSchemaType(field.name, fmap(field.name)), field.nullable))
    val hintedFieldNames = schemaHint.map(field => field.name)

    // Next, infer remaining fields.  Note that fmap does not declare/preserve any order
    val remaining = fmap.filterNot { case (name, feature) => hintedFieldNames.contains(name) }
    val remainingFields = remaining.toSeq.map { case (name, feature) =>
        StructField(name, _inferSchemaType(name, feature), nullable=true)
    }

    StructType(hintedFields ++ remainingFields)
  }

  /**
    * Converts a TensorFlow example to an equivalent Spark DataFrame Row.
    *
    * @param example TensorFlow Example
    * @param schema Fully-specified Spark schema
    * @return Spark DataFrame Row
    */
  def fromTFExample(example: Example, schema: StructType): Row = {
    import scala.collection.JavaConversions._
    val fields = schema.fields
    val fmap = example.getFeatures.getFeatureMap.toMap

    def _getValue(field: StructField): Any = {
      val f = fmap(field.name)
      f.getKindCase match {
        case KindCase.BYTES_LIST =>
          val byteList = f.getBytesList
          byteList.getValueCount match {
            case 0 => require(field.nullable, s"Field ${field.name} is not nullable"); null
            case 1 => field.dataType match {
              case BinaryType => byteList.getValue(0).toByteArray
              case BooleanType => byteList.getValue(0).byteAt(0) > 0
              case StringType => byteList.getValue(0).toStringUtf8
              case _ => throw new Exception(s"Unsupported BYTES_LIST type conversion to ${field.dataType}")
            }
            case _ => field.dataType match {
              case ArrayType(BinaryType,_) => byteList.getValueList.map(_.toByteArray).toList
              case ArrayType(BooleanType,_) => byteList.getValueList.map(_.byteAt(0) > 0).toList
              case ArrayType(StringType,_) => byteList.getValueList.map(_.toStringUtf8).toList
            }
          }
        case KindCase.INT64_LIST =>
          val int64List = f.getInt64List
          int64List.getValueCount match {
            case 0 => require(field.nullable, s"Field ${field.name} is not nullable"); null
            case 1 => field.dataType match {
              case BooleanType => int64List.getValue(0) != 0
              case IntegerType => int64List.getValue(0).toInt
              case LongType => int64List.getValue(0)
              case FloatType => int64List.getValue(0).toFloat
              case _ => throw new Exception(s"Unsupported INT64_LIST type conversion to ${field.dataType}")
            }
            case _ => field.dataType match {
              case ArrayType(BooleanType,_) => int64List.getValueList.map(_ != 0).toList
              case ArrayType(IntegerType,_) => int64List.getValueList.map(_.toInt).toList
              case ArrayType(LongType,_) => int64List.getValueList.toList
              case ArrayType(FloatType,_) => int64List.getValueList.map(_.toFloat).toList
              case _ => throw new Exception(s"Unsupported INT64_LIST conversion to ${field.dataType}")
            }
          }
        case KindCase.FLOAT_LIST =>
          val floatList = f.getFloatList
          floatList.getValueCount match {
            case 0 => require(field.nullable, s"Field ${field.name} is not nullable"); null
            case 1 => field.dataType match {
              case FloatType => floatList.getValue(0)
              case DoubleType => floatList.getValue(0).toDouble
              case _ => throw new Exception(s"Unsupported FLOAT_LIST conversion to ${field.dataType}")
            }
            case _ => field.dataType match {
              case ArrayType(FloatType, _) => floatList.getValueList.toList
              case ArrayType(DoubleType, _) => floatList.getValueList.map(_.toDouble).toList
              case _ => throw new Exception(s"Unsupported FLOAT_LIST conversion to ${field.dataType}")
            }
          }
        case KindCase.KIND_NOT_SET =>
          throw new Exception(s"Unsupported TFRecord type: ${f.getKindCase}")
      }
    }

    val cols = fields.map(_getValue)
    Row.fromSeq(cols)
  }

  def saveAsTFRecords(df: DataFrame, outputDir: String): Unit = {
    val rdd = df.rdd.mapPartitions(toTFExample(df.schema))
    val pairRDD = rdd.map(e => (new BytesWritable(e.toByteArray), null))
    pairRDD.saveAsNewAPIHadoopFile(outputDir,
                                  classOf[org.apache.hadoop.io.BytesWritable],
                                  classOf[org.apache.hadoop.io.NullWritable],
                                  classOf[org.tensorflow.hadoop.io.TFRecordFileOutputFormat])
  }

  def toTFExample(schema: StructType)(iter: Iterator[Row]): Iterator[Example] = {
    import scala.collection.JavaConversions._

    val dtypes = schema.fields.zipWithIndex.map { case (f,i) => (f.name, f.dataType, i) }

    def _toTFFeature(dtype: DataType, index: Int, row: Row): Feature = {
      val feature = dtype match {
        case BinaryType =>
          val bytes = row.getAs[Array[Byte]](index)
          Feature.newBuilder.setBytesList(BytesList.newBuilder.addValue(ByteString.copyFrom(bytes)))
        case BooleanType =>
          Feature.newBuilder.setInt64List(Int64List.newBuilder.addValue(if (row.getBoolean(index)) 1L else 0L))
        case FloatType =>
          Feature.newBuilder.setFloatList(FloatList.newBuilder.addValue(row.getFloat(index)))
        case DoubleType =>
          Feature.newBuilder.setFloatList(FloatList.newBuilder.addValue(row.getDouble(index).toFloat))
        case LongType =>
          Feature.newBuilder.setInt64List(Int64List.newBuilder.addValue(row.getLong(index)))
        case IntegerType =>
          Feature.newBuilder.setInt64List(Int64List.newBuilder.addValue(row.getInt(index)))
        case StringType =>
          Feature.newBuilder.setBytesList(BytesList.newBuilder.addValue(ByteString.copyFromUtf8(row.getString(index))))
        case ArrayType(BinaryType, _) =>
          val builder = BytesList.newBuilder
          row.getList[Array[Byte]](index).foreach(b => builder.addValue(ByteString.copyFrom(b)))
          Feature.newBuilder.setBytesList(builder)
        case ArrayType(BooleanType, _) =>
          val builder = Int64List.newBuilder
          row.getList[java.lang.Boolean](index).foreach(b => builder.addValue(if (b) 1L else 0L))
          Feature.newBuilder.setInt64List(builder)
        case ArrayType(IntegerType, _) =>
          val builder = Int64List.newBuilder
          row.getList[java.lang.Integer](index).foreach(i => builder.addValue(i.longValue))
          Feature.newBuilder.setInt64List(builder)
        case ArrayType(LongType, _) =>
          val arr = row.getList[java.lang.Long](index)
          Feature.newBuilder.setInt64List(Int64List.newBuilder.addAllValue(arr))
        case ArrayType(FloatType, _) =>
          val arr = row.getList[java.lang.Float](index)
          Feature.newBuilder.setFloatList(FloatList.newBuilder.addAllValue(arr))
        case ArrayType(DoubleType, _) =>
          val builder = FloatList.newBuilder
          row.getList[java.lang.Double](index).foreach(d => builder.addValue(d.floatValue))
          Feature.newBuilder.setFloatList(builder)
        case ArrayType(StringType, _) =>
          val arr = row.getList[String](index).toList.map(s => ByteString.copyFrom(s.getBytes))
          Feature.newBuilder.setBytesList(BytesList.newBuilder.addAllValue(arr))
      }
      feature.build()
    }

    var result = ListBuffer.empty[Example]
    for (row <- iter) {
      val fbuilder = Features.newBuilder()
      dtypes.foreach { case (name, dtype, index) =>
        fbuilder.putFeature(name, _toTFFeature(dtype, index, row))
      }
      val example = Example.newBuilder.setFeatures(fbuilder.build()).build()

      result += example
    }

    result.iterator
  }
}

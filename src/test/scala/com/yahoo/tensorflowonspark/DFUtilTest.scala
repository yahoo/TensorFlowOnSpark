/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter
import org.scalatest.Matchers._

import scala.collection.JavaConversions._

class DFUtilTest extends FunSuite with BeforeAndAfter with TestData {
  val conf: SparkConf = new SparkConf().setAppName("SparkInferTF").setMaster("local")
  implicit val sc: SparkContext = new SparkContext(conf)
  implicit val spark: SparkSession = SparkSession.builder.getOrCreate()

  before {
    FileUtils.deleteDirectory(new File("test-data"))
  }

  test("Save DataFrame as TFRecords and reload with same schema") {
    val df1 = spark.createDataFrame(List(row1, row2), schema)
    df1.show()
    df1.printSchema()
    assert(schema == df1.schema)

    // save to disk
    DFUtil.saveAsTFRecords(df1, "test-data")
    assert(new File("test-data").exists())

    // reload from disk
    val df2 = DFUtil.loadTFRecords("test-data", schema)
    df2.show()
    df2.printSchema()
    assert(df1.schema == df2.schema)

    // compare binary column
    val binaryIn = df1.select("binary").collect
    val binaryOut = df2.select("binary").collect
    assert(binaryOut(0).getAs[Array[Byte]](0) === binaryIn(0).getAs[Array[Byte]](0))
    assert(binaryOut(1).getAs[Array[Byte]](0) === binaryIn(1).getAs[Array[Byte]](0))

    // compare scalar columns
    val scalarsIn = df1.select("bool", "int", "long", "float", "double", "string").collect
    val scalarsOut = df2.select("bool", "int", "long", "float", "double", "string").collect

    assert(scalarsOut(0).toSeq == scalarsIn(0).toSeq)
    assert(scalarsOut(1).toSeq == scalarsIn(1).toSeq)

    // compare binary array column
    val binArraysIn = df1.select("arrayBinary").collect
    val binArraysOut = df2.select("arrayBinary").collect
    for (row <- 0 to 1) {
      val out = binArraysOut(row).getList[Array[Byte]](0)
      val in = binArraysIn(row).getList[Array[Byte]](0)
      for (i <- 0 to 2) {
        assert(out(i) === in(i))
      }
    }

    // compare array columms
    val arraysIn = df1.select("arrayBool", "arrayInt", "arrayLong", "arrayFloat", "arrayString").collect
    val arraysOut = df2.select("arrayBool", "arrayInt", "arrayLong", "arrayFloat", "arrayString").collect

    assert(arraysOut(0).toSeq == arraysIn(0).toSeq)
    assert(arraysOut(1).toSeq == arraysIn(1).toSeq)

    assert(arraysOut(0).getList[Boolean](0) === arraysIn(0).getList[Boolean](0))
    assert(arraysOut(0).getList[Int](1) === arraysIn(0).getList[Int](1))
    assert(arraysOut(0).getList[Long](2) === arraysIn(0).getList[Long](2))
    assert(arraysOut(0).getList[Float](3) === arraysIn(0).getList[Float](3))

    // compare arrayDouble columns
    // Note: there is loss of precision since we convert double => float when saving,
    // and then convert float => double when loading.  So need to use epsilon comparison.
    val arrayDoubleIn = df1.select("arrayDouble").collect
    val arrayDoubleOut = df2.select("arrayDouble").collect
    for (row <- 0 to 1) {
      val out = arrayDoubleOut(row).getList[Double](0)
      val in = arrayDoubleIn(row).getList[Double](0)
      for (i <- 0 to 2) {
        assert(out(i) === in(i) +- 1e-6)
      }
    }
  }

  test("Save DataFrame as TFRecords and reload without schema") {
    val df1 = spark.createDataFrame(List(row1, row2), schema)
    df1.show()
    df1.printSchema()
    assert(schema == df1.schema)

    // save to disk
    DFUtil.saveAsTFRecords(df1, "test-data")
    assert(new File("test-data").exists())

    // reload from disk w/o schema hint
    val df2 = DFUtil.loadTFRecords("test-data")
    df2.show()
    df2.printSchema()

    // convert schema to list of StructFields, sorted by name
    val actual = df2.schema.fields.sortBy(_.name)

    // this is the expected inferred StructFields, sorted by name
    val expected = Array(
      StructField("binary", StringType),
      StructField("bool", LongType),
      StructField("int", LongType),
      StructField("long", LongType),
      StructField("float", FloatType),
      StructField("double", FloatType),
      StructField("string", StringType),
      StructField("arrayBinary", ArrayType(StringType)),
      StructField("arrayBool", ArrayType(LongType)),
      StructField("arrayInt", ArrayType(LongType)),
      StructField("arrayLong", ArrayType(LongType)),
      StructField("arrayFloat", ArrayType(FloatType)),
      StructField("arrayDouble", ArrayType(FloatType)),
      StructField("arrayString", ArrayType(StringType))
    ).sortBy(_.name)

    assert(actual === expected)
  }
}

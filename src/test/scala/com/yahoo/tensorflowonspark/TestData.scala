/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

trait TestData {
  val row1 = Row("one".getBytes, true, 1, 1L, 1.0f, 1.0, "one",
    Seq[Array[Byte]]("one".getBytes, "two".getBytes, "three".getBytes),
    Seq[Boolean](true, true, true),
    Seq[Int](1, 2, 3),
    Seq[Long](1L, 2L, 3L),
    Seq[Float](1.0f, 1.1f, 1.2f),
    Seq[Double](1.0, 1.1, 1.2),
    Seq[String]("one", "two", "three"))
  val row2 = Row("foo".getBytes, false, 2, 2L, 2.0f, 2.0, "foo",
    Seq[Array[Byte]]("foo".getBytes, "bar".getBytes, "baz".getBytes),
    Seq[Boolean](false, false, false),
    Seq[Int](4, 5, 6),
    Seq[Long](4L, 5L, 6L),
    Seq[Float](2.0f, 2.1f, 2.2f),
    Seq[Double](2.0, 2.1, 2.2),
    Seq[String]("foo", "bar", "baz"))

  val listRows = List(row1, row2)

  val schema = StructType(Array(
    StructField("binary", BinaryType),
    StructField("bool", BooleanType),
    StructField("int", IntegerType),
    StructField("long", LongType),
    StructField("float", FloatType),
    StructField("double", DoubleType),
    StructField("string", StringType),
    StructField("arrayBinary", ArrayType(BinaryType)),
    StructField("arrayBool", ArrayType(BooleanType)),
    StructField("arrayInt", ArrayType(IntegerType)),
    StructField("arrayLong", ArrayType(LongType)),
    StructField("arrayFloat", ArrayType(FloatType)),
    StructField("arrayDouble", ArrayType(DoubleType)),
    StructField("arrayString", ArrayType(StringType))
  ))
}

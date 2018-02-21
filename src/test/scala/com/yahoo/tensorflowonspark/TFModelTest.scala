/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import java.nio._

import org.apache.spark.sql.Row
import org.scalatest.FunSuite
import org.tensorflow.{DataType, Tensor}


class TFModelTest extends FunSuite with TestData {
  val model = new TFModel("test")

  test("Convert Rows to Tensors") {
    val tensors = model.batch2tensors(listRows, schema)

    // given 2 rows of M columns in listRows
    // expect M tensors with 2 rows each, with ArrayType tensors having 3 cols each
    assert(tensors.size == listRows.head.size)
    assert(tensors.forall { case (name, tensor) =>
      val expectedShape = if (name.startsWith("array")) Array(2L, 3L) else Array(2L)
      tensor.shape() sameElements expectedShape })

    // check "sum" of columns for numeric scalar types
    assert(tensors("bool").copyTo(Array.ofDim[Boolean](2)) === Array(true, false))
    assert(tensors("int").copyTo(Array.ofDim[Int](2)).sum === 3)
    assert(tensors("long").copyTo(Array.ofDim[Long](2)).sum === 3L)
    assert(tensors("float").copyTo(Array.ofDim[Float](2)).sum === 3.0f)
    assert(tensors("double").copyTo(Array.ofDim[Double](2)).sum === 3.0)

    // check binary/string types
    assert(tensors("binary").copyTo(Array.ofDim[Array[Byte]](2)) === Array("one".getBytes, "foo".getBytes))
    assert(tensors("string").copyTo(Array.ofDim[Array[Byte]](2)).map(new String(_)) === Array("one", "foo"))

    // check sum of rows for numeric array types
    assert(tensors("arrayBool").copyTo(Array.ofDim[Boolean](2,3)).map(row => row.reduce((x,y) => x && y)) === Array(true, false))
    assert(tensors("arrayInt").copyTo(Array.ofDim[Int](2,3)).map(_.sum) === Array(6, 15))
    assert(tensors("arrayLong").copyTo(Array.ofDim[Long](2,3)).map(_.sum) === Array(6L, 15L))
    assert(tensors("arrayFloat").copyTo(Array.ofDim[Float](2,3)).map(_.sum) === Array(3.3f, 6.3f))
    assert(tensors("arrayDouble").copyTo(Array.ofDim[Double](2,3)).map(_.sum) === Array(3.3, 6.3))

    // check binary/string array types
    assert(tensors("arrayBinary").copyTo(Array.ofDim[Array[Byte]](2, 3)) ===
      Array(
        Array("one".getBytes, "two".getBytes, "three".getBytes),
        Array("foo".getBytes, "bar".getBytes, "baz".getBytes)
      ))
    assert(tensors("arrayString").copyTo(Array.ofDim[Array[Byte]](2, 3)).map(row =>
      row.map(new String(_))) ===
      Array(
        Array("one", "two", "three"),
        Array("foo", "bar", "baz")
      )
    )
  }

  test("Convert 1D Tensor to Rows") {
    val floatBuf = FloatBuffer.wrap(Array(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f))
    val t8 = Tensor.create(Array(8L), floatBuf)
    val rows: List[Row] = model.tensors2batch(Seq(t8))
    assert(rows(0).getAs[Float](0) == 0.0f)
    assert(rows(1).getAs[Float](0) == 1.0f)
    assert(rows(2).getAs[Float](0) == 2.0f)
    assert(rows(3).getAs[Float](0) == 3.0f)
  }

  test("Convert 2D Tensor to Rows") {
    // 8 x 1 tensor
    val floatBuf = FloatBuffer.wrap(Array(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f))
    val t8_1 = Tensor.create(Array(8L, 1L), floatBuf)
    var rows: List[Row] = model.tensors2batch(Seq(t8_1))
    assert(rows(0).getAs[Array[Float]](0).sum == 0.0f)
    assert(rows(1).getAs[Array[Float]](0).sum == 1.0f)
    assert(rows(2).getAs[Array[Float]](0).sum == 2.0f)
    assert(rows(3).getAs[Array[Float]](0).sum == 3.0f)

    // 4 x 2 tensor
    floatBuf.rewind()
    val t4_2 = Tensor.create(Array(4L, 2L), floatBuf)
    rows = model.tensors2batch(Seq(t4_2))
    assert(rows(0).getAs[Array[Float]](0).sum == 1.0f)
    assert(rows(1).getAs[Array[Float]](0).sum == 5.0f)
    assert(rows(2).getAs[Array[Float]](0).sum == 9.0f)
    assert(rows(3).getAs[Array[Float]](0).sum == 13.0f)

    // 2 x 4 tensor
    floatBuf.rewind()
    val t2_4 = Tensor.create(Array(2L, 4L), floatBuf)
    rows = model.tensors2batch(Seq(t2_4))
    assert(rows(0).getAs[Array[Float]](0).sum == 6.0f)
    assert(rows(1).getAs[Array[Float]](0).sum == 22.0f)

    // 1 x 8 tensor
    floatBuf.rewind()
    val t1_8 = Tensor.create(Array(1L, 8L), floatBuf)
    rows = model.tensors2batch(Seq(t1_8))
    assert(rows(0).getAs[Array[Float]](0).sum == 28.0f)
  }

  test("Convert multiple Tensors to Rows") {
    // 1D tensor
    val longBuf = LongBuffer.wrap(Array(0L, 1L, 2L, 3L))
    val t4 = Tensor.create(Array(4L), longBuf)

    // 4 x 2 tensor
    val floatBuf = FloatBuffer.wrap(Array(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f))
    val t4_2 = Tensor.create(Array(4L, 2L), floatBuf)

    val rows = model.tensors2batch(Seq(t4, t4_2))

    // expect Rows:
    // 0L, (0.0f, 1.0f)
    // 1L, (2.0f, 3.0f)
    // 2L, (4.0f, 5.0f)
    // 3L, (6.0f, 7.0f)
    assert(rows(0).getLong(0) == 0L)
    assert(rows(0).getAs[Array[Float]](1).sum == 1.0f)
    assert(rows(1).getLong(0) == 1L)
    assert(rows(1).getAs[Array[Float]](1).sum == 5.0f)
    assert(rows(2).getLong(0) == 2L)
    assert(rows(2).getAs[Array[Float]](1).sum == 9.0f)
    assert(rows(3).getLong(0) == 3L)
    assert(rows(3).getAs[Array[Float]](1).sum == 13.0f)
  }
}
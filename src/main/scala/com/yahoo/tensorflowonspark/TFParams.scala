/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.apache.spark.ml.param.{Param, Params}

trait TFParams extends Params {

  final val batchSize: Param[Int] = new Param[Int](this, "batchSize",
    "Batch size for consuming input data.  Default: 128")
  final def getBatchSize: Int = $(batchSize)
  final def setBatchSize(i: Int): this.type = set(batchSize, i)
  setDefault(batchSize, 128)

  final val model: Param[String] = new Param[String](this, "model",
    "Path to TensorFlow saved_model file")
  final def getModel: String = $(model)
  final def setModel(s: String): this.type = set(model, s)
  setDefault(model, "")

  final val tag: Param[String] = new Param[String](this, "tag",
    "String tag for graph within model.  Default: \"serve\"")
  final def getTag: String = $(tag)
  final def setTag(s: String): this.type = set(tag, s)
  setDefault(tag, "serve")

  final val inputMapping: Param[Map[String, String]] = new Param[Map[String, String]](this, "inputMapping",
    "mapping of input DataFrame column name to TensorFlow input tensor name")
  final def getInputMapping: Map[String, String] = $(inputMapping)
  final def setInputMapping(m: Map[String, String]): this.type = set(inputMapping, m)
  setDefault(inputMapping, Map.empty[String, String])

  final val outputMapping: Param[Map[String, String]] = new Param[Map[String, String]](this, "outputMapping",
    "mapping of TensorFlow output tensor name to output DataFrame column name")
  final def getOutputMapping: Map[String, String] = $(outputMapping)
  final def setOutputMapping(m: Map[String, String]): this.type = set(outputMapping, m)
  setDefault(outputMapping, Map.empty[String, String])


}

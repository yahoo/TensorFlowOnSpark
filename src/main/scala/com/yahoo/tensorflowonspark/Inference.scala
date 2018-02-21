/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s._
import org.json4s.native.JsonMethods

/**
  * Spark application that infers from a TensorFlow SavedModel.
  */
object Inference {

  case class Config(export_dir: String = "",
                    input: String = "",
                    schema_hint: StructType = new StructType(),
                    input_mapping: Map[String, String] = Map.empty,
                    output_mapping: Map[String, String] = Map.empty,
                    output: String = "",
                    verbose: Boolean = false)

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Inference")
    implicit val sc: SparkContext = new SparkContext(conf)
    val parser = new scopt.OptionParser[Config]("Inference") {
      opt[String]("export_dir").text("Path to exported saved_model")
        .action((x, conf) => conf.copy(export_dir = x))
      opt[String]("input").text("Path to input TFRecords")
        .action((x, conf) => conf.copy(input = x))
      opt[String]("schema_hint").text("schema hint (in StructType.simpleString format) for converting TFRecord features to Spark DataFrame types")
        .action{ case (schema, conf) => conf.copy(schema_hint = SimpleTypeParser.parse(schema)) }
      opt[String]("input_mapping").text("JSON mapping of input columns to input tensors")
        .action((x, conf) => conf.copy(input_mapping = JsonMethods.parse(x).values.asInstanceOf[Map[String, String]]))
      opt[String]("output_mapping").text("JSON mapping of output tensors to output columns")
        .action((x, conf) => conf.copy(output_mapping = JsonMethods.parse(x).values.asInstanceOf[Map[String, String]]))
      opt[String]("output").text("Path to write predictions").action((x, conf) => conf.copy(output = x))
      opt[Unit]("verbose").text("Print input dataframe sample with schema").action((_, conf) => conf.copy(verbose = true))
    }

    parser.parse(args, Config()) match {
      case Some(config) => run(sc, config)
      case None => System.exit(1)
    }
    sc.stop()
  }

  def run(implicit sc: SparkContext, config: Config) {

    implicit val spark: SparkSession = SparkSession.builder().getOrCreate()

    // load TFRecords as a Spark DataFrame (using a user-provided schema hint)
    val df = DFUtil.loadTFRecords(config.input, config.schema_hint)
    if (config.verbose) {
      df.show()
      df.printSchema()
    }

    // instantiate a TFModel pointing to an existing TensorFlow saved_model export
    // set up mappings between input DataFrame columns to input Tensors
    // and output Tensors to output DataFrame columns
    // Note: the output DataFrame column types will be inferred from the output Tensor dtypes
    val model = new TFModel().setModel(config.export_dir)
                              .setInputMapping(config.input_mapping)
                              .setOutputMapping(config.output_mapping)

    // transform the input DataFrame
    // Note: we're currently dropping input columns for simplicity, you can retrieve them as Tensors if needed.
    val predDF = model.transform(df)

    // write the predictions
    predDF.write.json(config.output)

    spark.stop()
  }
}

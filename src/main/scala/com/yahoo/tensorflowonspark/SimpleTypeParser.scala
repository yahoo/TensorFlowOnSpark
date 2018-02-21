/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.apache.spark.sql.types._

import scala.util.parsing.combinator.RegexParsers

/**
  * Parser which generates a StructType from a string of StructType.simpleString format.
  *
  * Currently, this supports the following base types:
  * - binary
  * - boolean
  * - int
  * - long (not a simpleString keyword, but provided here for ease of use)
  * - bigint
  * - float
  * - double
  * - string
  *
  * Additionally, this supports single-dimensional arrays of the base types.
  */
object SimpleTypeParser {
  def parse(simpleString: String): StructType = {
    val parser = new SimpleTypeParser
    parser.parseAll(parser.struct, simpleString).get
  }
}

class SimpleTypeParser extends RegexParsers {
  val name = "[a-zA-Z][/a-zA-Z_-]*".r

  def baseType: Parser[DataType] = ("binary" | "boolean" | "int" | "long" | "bigint" | "float" | "double" | "string") ^^ {
    case "binary" => BinaryType
    case "boolean" => BooleanType
    case "int" => IntegerType
    case "long" => LongType
    case "bigint" => LongType
    case "float" => FloatType
    case "double" => DoubleType
    case "string" => StringType
  }

  def arrayType: Parser[DataType] = ("array<" ~ baseType ~ ">") ^^ {
    case "array<" ~ bt ~ ">" => ArrayType(bt)
  }

  def dataType: Parser[DataType] = baseType | arrayType

  def field: Parser[StructField] = (name ~ ":" ~ dataType) ^^ {
    case n ~ ":" ~ t => StructField(n, t)
  }
  def fieldList: Parser[Seq[StructField]] = (field ~ opt("," ~ fieldList)) ^^ {
    case f ~ None => Seq(f)
    case f ~ Some("," ~ fl) => f +: fl
  }
  def struct: Parser[StructType] = ("struct<" ~ fieldList ~ ">") ^^ {
    case "struct<" ~ fl ~ ">" => StructType(fl)
  }
}

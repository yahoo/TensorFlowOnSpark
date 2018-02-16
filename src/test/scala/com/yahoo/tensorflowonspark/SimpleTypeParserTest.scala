package com.yahoo.tensorflowonspark

import org.scalatest.FunSuite

class SimpleTypeParserTest extends FunSuite with TestData {
  test("parse simple type string as schema") {
    val s = schema.simpleString
    val parsed = SimpleTypeParser.parse(s)
    assert(parsed == schema)
  }
}

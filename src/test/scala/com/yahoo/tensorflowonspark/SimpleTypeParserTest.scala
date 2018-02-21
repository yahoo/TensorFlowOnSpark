/**
  * Copyright 2018 Yahoo Inc.
  * Licensed under the terms of the Apache 2.0 license.
  * Please see LICENSE file in the project root for terms.
  */
package com.yahoo.tensorflowonspark

import org.scalatest.FunSuite

class SimpleTypeParserTest extends FunSuite with TestData {
  test("parse simple type string as schema") {
    val s = schema.simpleString
    val parsed = SimpleTypeParser.parse(s)
    assert(parsed == schema)
  }
}

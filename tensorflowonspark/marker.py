# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

class Marker(object):
  """Base class for special marker objects in the data queue"""
  pass

class EndPartition(Marker):
  """Marks the end of an RDD Partition during data feeding"""
  pass


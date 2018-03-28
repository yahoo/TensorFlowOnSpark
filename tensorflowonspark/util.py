# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import os
import socket

def get_ip_address():
  """Simple utility to get host IP address."""
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  return s.getsockname()[0]


def find_in_path(path, file):
  """Find a file in a given path string."""
  for p in path.split(os.pathsep):
    candidate = os.path.join(p, file)
    if os.path.exists(candidate) and os.path.isfile(candidate):
      return candidate
  return False


def write_executor_id(num):
  """Write executor_id into a local file in the executor's current working directory"""
  with open("executor_id", "w") as f:
    f.write(str(num))


def read_executor_id():
  """Read worker id from a local file in the executor's current working directory"""
  with open("executor_id", "r") as f:
    return int(f.read())

# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import os
import socket
import subprocess
import errno
from socket import error as socket_error
from . import gpu_info


def single_node_env(num_gpus=1):
  """Setup environment variables for Hadoop compatibility and GPU allocation"""
  import tensorflow as tf
  # ensure expanded CLASSPATH w/o glob characters (required for Spark 2.1 + JNI)
  if 'HADOOP_PREFIX' in os.environ and 'TFOS_CLASSPATH_UPDATED' not in os.environ:
      classpath = os.environ['CLASSPATH']
      hadoop_path = os.path.join(os.environ['HADOOP_PREFIX'], 'bin', 'hadoop')
      hadoop_classpath = subprocess.check_output([hadoop_path, 'classpath', '--glob']).decode()
      os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath
      os.environ['TFOS_CLASSPATH_UPDATED'] = '1'

  # reserve GPU, if requested
  if tf.test.is_built_with_cuda():
    gpus_to_use = gpu_info.get_gpus(num_gpus)
    logging.info("Using gpu(s): {0}".format(gpus_to_use))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
  else:
    # CPU
    logging.info("Using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_ip_address():
  """Simple utility to get host IP address."""
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
  except socket_error as sockerr:
    if sockerr.errno != errno.ENETUNREACH:
      raise sockerr
    ip_address = socket.gethostbyname(socket.getfqdn())
  finally:
    s.close()

  return ip_address


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

# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import random
import subprocess
import time

logger = logging.getLogger(__name__)

MAX_RETRIES = 3           #: Maximum retries to allocate GPUs
AS_STRING = 'string'
AS_LIST = 'list'


def is_gpu_available():
  """Determine if GPUs are available on the host"""
  try:
    subprocess.check_output(["nvidia-smi", "--list-gpus"])
    return True
  except Exception:
    return False


def get_gpus(num_gpu=1, worker_index=-1, format=AS_STRING):
  """Get list of free GPUs according to nvidia-smi.

  This will retry for ``MAX_RETRIES`` times until the requested number of GPUs are available.

  Args:
    :num_gpu: number of GPUs desired.
    :worker_index: index "hint" for allocation of available GPUs.

  Returns:
    Comma-delimited string of GPU ids, or raises an Exception if the requested number of GPUs could not be found.
  """
  # get list of gpus (index, uuid)
  list_gpus = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode()
  logger.debug("all GPUs:\n{0}".format(list_gpus))

  # parse index and guid
  gpus = [x for x in list_gpus.split('\n') if len(x) > 0]

  def parse_gpu(gpu_str):
    cols = gpu_str.split(' ')
    return cols[5].split(')')[0], cols[1].split(':')[0]

  gpu_list = [parse_gpu(gpu) for gpu in gpus]

  free_gpus = []
  retries = 0
  while len(free_gpus) < num_gpu and retries < MAX_RETRIES:
    smi_output = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits", "--query-compute-apps=gpu_uuid"]).decode()
    logger.debug("busy GPUs:\n{0}".format(smi_output))
    busy_uuids = [x for x in smi_output.split('\n') if len(x) > 0]
    for uuid, index in gpu_list:
      if uuid not in busy_uuids:
        free_gpus.append(index)

    if len(free_gpus) < num_gpu:
      logger.warn("Unable to find available GPUs: requested={0}, available={1}".format(num_gpu, len(free_gpus)))
      retries += 1
      time.sleep(30 * retries)
      free_gpus = []

  logger.info("Available GPUs: {}".format(free_gpus))

  # if still can't find available GPUs, raise exception
  if len(free_gpus) < num_gpu:
    smi_output = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"]).decode()
    logger.info(": {0}".format(smi_output))
    raise Exception("Unable to find {} free GPU(s)\n{}".format(num_gpu, smi_output))

  # Get logical placement
  num_available = len(free_gpus)
  if worker_index == -1:
    # use original random placement
    random.shuffle(free_gpus)
    proposed_gpus = free_gpus[:num_gpu]
  else:
    # ordered by worker index
    if worker_index * num_gpu + num_gpu > num_available:
      worker_index = worker_index * num_gpu % num_available
    proposed_gpus = free_gpus[worker_index * num_gpu:(worker_index * num_gpu + num_gpu)]
  logger.info("Proposed GPUs: {}".format(proposed_gpus))

  if format == AS_STRING:
    return ','.join(str(x) for x in proposed_gpus)
  elif format == AS_LIST:
    return proposed_gpus
  else:
    raise Exception("Unknown GPU format")

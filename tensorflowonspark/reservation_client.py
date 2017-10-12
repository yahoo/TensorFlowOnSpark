# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""
Simple utility to shutdown a Spark StreamingContext by signaling the reservation Server.
Note: use the reservation server address (host, port) reported in the driver logs.
"""

import reservation
import sys

if __name__ == "__main__":
  host = sys.argv[1]
  port = int(sys.argv[2])
  addr = (host, port)
  client = reservation.Client(addr)
  client.request_stop()
  client.close()

import reservation
import sys

"""
Simple utility to shutdown a Spark StreamingContext by signaling the reservation Server
Note: use the reservation server (host, port) reported in the driver logs.
"""
host = sys.argv[1]
port = int(sys.argv[2])
addr = (host, port)
client = reservation.Client(addr)
client.request_stop()
client.close()

# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module contains client/server methods to manage node reservations during TFCluster startup."""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import os
import pickle
import select
import socket
import struct
import sys
import threading
import time

from . import util

logger = logging.getLogger(__name__)

TFOS_SERVER_PORT = "TFOS_SERVER_PORT"
TFOS_SERVER_HOST = "TFOS_SERVER_HOST"
BUFSIZE = 1024
MAX_RETRIES = 3


class Reservations:
  """Thread-safe store for node reservations.

  Args:
    :required: expected number of nodes in the cluster.
  """

  def __init__(self, required):
    self.required = required
    self.lock = threading.RLock()
    self.reservations = []

  def add(self, meta):
    """Add a reservation.

    Args:
      :meta: a dictonary of metadata about a node
    """
    with self.lock:
      self.reservations.append(meta)

  def done(self):
    """Returns True if the ``required`` number of reservations have been fulfilled."""
    with self.lock:
      return len(self.reservations) >= self.required

  def get(self):
    """Get the list of current reservations."""
    with self.lock:
      return self.reservations

  def remaining(self):
    """Get a count of remaining/unfulfilled reservations."""
    with self.lock:
      return self.required - len(self.reservations)


class MessageSocket(object):
  """Abstract class w/ length-prefixed socket send/receive functions."""

  def receive(self, sock):
    """Receive a message on ``sock``."""
    msg = None
    data = b''
    recv_done = False
    recv_len = -1
    while not recv_done:
      buf = sock.recv(BUFSIZE)
      if buf is None or len(buf) == 0:
        raise Exception("socket closed")
      if recv_len == -1:
        recv_len = struct.unpack('>I', buf[:4])[0]
        data += buf[4:]
        recv_len -= len(data)
      else:
        data += buf
        recv_len -= len(buf)
      recv_done = (recv_len == 0)

    msg = pickle.loads(data)
    return msg

  def send(self, sock, msg):
    """Send ``msg`` to destination ``sock``."""
    data = pickle.dumps(msg)
    buf = struct.pack('>I', len(data)) + data
    sock.sendall(buf)


class Server(MessageSocket):
  """Simple socket server with length-prefixed pickle messages.

  Args:
    :count: expected number of nodes in the cluster.
  """
  reservations = None             #: List of reservations managed by this server.
  done = False                    #: boolean indicating if server should be shutdown.

  def __init__(self, count):
    assert count > 0, "Expected number of reservations should be greater than zero"
    self.reservations = Reservations(count)

  def await_reservations(self, sc, status={}, timeout=600):
    """Block until all reservations are received."""
    timespent = 0
    while not self.reservations.done():
      logger.info("waiting for {0} reservations".format(self.reservations.remaining()))
      # check status flags for any errors
      if 'error' in status:
        sc.cancelAllJobs()
        sc.stop()
        sys.exit(1)
      time.sleep(1)
      timespent += 1
      if (timespent > timeout):
        raise Exception("timed out waiting for reservations to complete")
    logger.info("all reservations completed")
    return self.reservations.get()

  def _handle_message(self, sock, msg):
    logger.debug("received: {0}".format(msg))
    msg_type = msg['type']
    if msg_type == 'REG':
      self.reservations.add(msg['data'])
      MessageSocket.send(self, sock, 'OK')
    elif msg_type == 'QUERY':
      MessageSocket.send(self, sock, self.reservations.done())
    elif msg_type == 'QINFO':
      rinfo = self.reservations.get()
      MessageSocket.send(self, sock, rinfo)
    elif msg_type == 'STOP':
      logger.info("setting server.done")
      MessageSocket.send(self, sock, 'OK')
      self.done = True
    else:
      MessageSocket.send(self, sock, 'ERR')

  def start(self):
    """Start listener in a background thread

    Returns:
      address of the Server as a tuple of (host, port)
    """
    server_sock = self.start_listening_socket()

    # hostname may not be resolvable but IP address probably will be
    host = self.get_server_ip()
    port = server_sock.getsockname()[1]
    addr = (host, port)
    logger.info("listening for reservations at {0}".format(addr))

    def _listen(self, sock):
      CONNECTIONS = []
      CONNECTIONS.append(sock)

      while not self.done:
        read_socks, write_socks, err_socks = select.select(CONNECTIONS, [], [], 60)
        for sock in read_socks:
          if sock == server_sock:
            client_sock, client_addr = sock.accept()
            CONNECTIONS.append(client_sock)
            logger.debug("client connected from {0}".format(client_addr))
          else:
            try:
              msg = self.receive(sock)
              self._handle_message(sock, msg)
            except Exception as e:
              logger.debug(e)
              sock.close()
              CONNECTIONS.remove(sock)

      server_sock.close()

    t = threading.Thread(target=_listen, args=(self, server_sock))
    t.daemon = True
    t.start()

    return addr

  def get_server_ip(self):
    """Returns the value of TFOS_SERVER_HOST environment variable (if set), otherwise defaults to current host/IP."""
    return os.getenv(TFOS_SERVER_HOST, util.get_ip_address())

  def get_server_ports(self):
    """Returns a list of target ports as defined in the TFOS_SERVER_PORT environment (if set), otherwise defaults to 0 (any port).

    TFOS_SERVER_PORT should be either a single port number or a range, e.g. '8888' or '9997-9999'
    """
    port_string = os.getenv(TFOS_SERVER_PORT, "0")
    if '-' not in port_string:
      return [int(port_string)]
    else:
      ports = port_string.split('-')
      if len(ports) != 2:
        raise Exception("Invalid TFOS_SERVER_PORT: {}".format(port_string))
      return list(range(int(ports[0]), int(ports[1]) + 1))

  def start_listening_socket(self):
    """Starts the registration server socket listener."""
    port_list = self.get_server_ports()
    for port in port_list:
      try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('', port))
        server_sock.listen(10)
        logger.info("Reservation server binding to port {}".format(port))
        break
      except Exception as e:
        logger.warn("Unable to bind to port {}, error {}".format(port, e))
        server_sock = None
        pass

    if not server_sock:
      raise Exception("Reservation server unable to bind to any ports, port_list = {}".format(port_list))

    return server_sock

  def stop(self):
    """Stop the Server's socket listener."""
    self.done = True


class Client(MessageSocket):
  """Client to register and await node reservations.

  Args:
    :server_addr: a tuple of (host, port) pointing to the Server.
  """
  sock = None                   #: socket to server TCP connection
  server_addr = None            #: address of server

  def __init__(self, server_addr):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(server_addr)
    self.server_addr = server_addr
    logger.info("connected to server at {0}".format(server_addr))

  def _request(self, msg_type, msg_data=None):
    """Helper function to wrap msg w/ msg_type."""
    msg = {}
    msg['type'] = msg_type
    if msg_data:
      msg['data'] = msg_data

    done = False
    tries = 0
    while not done and tries < MAX_RETRIES:
      try:
        MessageSocket.send(self, self.sock, msg)
        done = True
      except socket.error as e:
        tries += 1
        if tries >= MAX_RETRIES:
          raise
        print("Socket error: {}".format(e))
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)

    logger.debug("sent: {0}".format(msg))
    resp = MessageSocket.receive(self, self.sock)
    logger.debug("received: {0}".format(resp))
    return resp

  def close(self):
    """Close the client socket."""
    self.sock.close()

  def register(self, reservation):
    """Register ``reservation`` with server."""
    resp = self._request('REG', reservation)
    return resp

  def get_reservations(self):
    """Get current list of reservations."""
    cluster_info = self._request('QINFO')
    return cluster_info

  def await_reservations(self):
    """Poll until all reservations completed, then return cluster_info."""
    done = False
    while not done:
      done = self._request('QUERY')
      time.sleep(1)
    return self.get_reservations()

  def request_stop(self):
    """Request server stop."""
    resp = self._request('STOP')
    return resp

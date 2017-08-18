# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import pickle
import select
import socket
import struct
import threading
import time

from . import util

BUFSIZE = 1024
MAX_RETRIES = 3

class Reservations:

  def __init__(self, required):
    self.required = required
    self.lock = threading.RLock()
    self.reservations = []

  def add(self, meta):
    with self.lock:
      self.reservations.append(meta)

  def done(self):
    with self.lock:
      return len(self.reservations) >= self.required

  def get(self):
    with self.lock:
      return self.reservations

  def remaining(self):
    with self.lock:
      return self.required - len(self.reservations)

class MessageSocket(object):
  """Abstract class w/ length-prefixed socket send/receive functions"""

  def receive(self, sock):
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
    data = pickle.dumps(msg)
    buf = struct.pack('>I', len(data)) + data
    sock.sendall(buf)


class Server(MessageSocket):
  """Simple socket server with length prefixed pickle messages"""
  reservations = None
  done = False

  def __init__(self, count):
    assert count > 0
    self.reservations = Reservations(count)

  def await_reservations(self):
    """Block until all reservations done"""
    while not self.reservations.done():
      logging.info("waiting for {0} reservations".format(self.reservations.remaining()))
      time.sleep(1)
    logging.info("all reservations completed")
    return self.reservations.get()

  def handle_message(self, sock, msg):
    logging.debug("received: {0}".format(msg))
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
      logging.info("setting server.done")
      MessageSocket.send(self, sock, 'OK')
      self.done = True
    else:
      MessageSocket.send(self, sock, 'ERR')

  def start(self):
    """Start listener in a background thread"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('',0))
    server_sock.listen(10)

    # hostname may not be resolvable but IP address probably will be
    host = util.get_ip_address()
    port = server_sock.getsockname()[1]
    addr = (host,port)
    logging.info("listening for reservations at {0}".format(addr))

    def _listen(self, sock):
      CONNECTIONS = []
      CONNECTIONS.append(sock)

      while not self.done:
        read_socks, write_socks, err_socks = select.select(CONNECTIONS, [], [], 60)
        for sock in read_socks:
          if sock == server_sock:
            client_sock, client_addr = sock.accept()
            CONNECTIONS.append(client_sock)
            logging.debug("client connected from {0}".format(client_addr))
          else:
            try:
              msg = self.receive(sock)
              self.handle_message(sock, msg)
            except Exception as e:
              logging.debug(e)
              sock.close()
              CONNECTIONS.remove(sock)

      server_sock.close()

    t = threading.Thread(target=_listen, args=(self, server_sock))
    t.daemon = True
    t.start()

    return addr

  def stop(self):
    """Stop the socket listener"""
    self.done = True

class Client(MessageSocket):
  """Client to register/await node reservations"""
  sock = None
  server_addr = None

  def __init__(self, server_addr):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(server_addr)
    self.server_addr = server_addr
    logging.info("connected to server at {0}".format(server_addr))

  def _request(self, msg_type, msg_data=None):
    """Helper function to wrap msg w/ msg_type"""
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

    logging.debug("sent: {0}".format(msg))
    resp = MessageSocket.receive(self, self.sock)
    logging.debug("received: {0}".format(resp))
    return resp

  def close(self):
    self.sock.close()

  def register(self, reservation):
    """Register reservation with server"""
    resp = self._request('REG', reservation)
    return resp

  def get_reservations(self):
    """Get current list of reservations"""
    cluster_info = self._request('QINFO')
    return cluster_info

  def await_reservations(self):
    """Poll until all reservations completed, then return cluster_info"""
    done = False
    while not done:
      done = self._request('QUERY')
      time.sleep(1)
    return self.get_reservations()

  def request_stop(self):
    resp = self._request('STOP')
    return resp

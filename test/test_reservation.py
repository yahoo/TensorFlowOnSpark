import os
import sys
import threading
import time
import unittest

from tensorflowonspark import util
from tensorflowonspark.reservation import Reservations, Server, Client

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock

class ReservationTest(unittest.TestCase):
  def test_reservation_class(self):
    """Test core reservation class, expecting 2 reservations"""
    r = Reservations(2)
    self.assertFalse(r.done())

    # add first reservation
    r.add({'node': 1})
    self.assertFalse(r.done())
    self.assertEqual(r.remaining(), 1)

    # add second reservation
    r.add({'node': 2})
    self.assertTrue(r.done())
    self.assertEqual(r.remaining(), 0)

    # get final list
    reservations = r.get()
    self.assertEqual(len(reservations), 2)

  def test_reservation_server(self):
    """Test reservation server, expecting 1 reservation"""
    s = Server(1)
    addr = s.start()

    # add first reservation
    c = Client(addr)
    resp = c.register({'node': 1})
    self.assertEqual(resp, 'OK')

    # get list of reservations
    reservations = c.get_reservations()
    self.assertEqual(len(reservations), 1)

    # should return immediately with list of reservations
    reservations = c.await_reservations()
    self.assertEqual(len(reservations), 1)

    # request server stop
    c.request_stop()
    time.sleep(1)
    self.assertEqual(s.done, True)

  def test_reservation_enviroment_exists_get_server_ip_return_environment_value(self):
      tfso_server = Server(5)
      with mock.patch.dict(os.environ,{'TFOS_SERVER_HOST':'my_host_ip'}):
        assert tfso_server.get_server_ip() == "my_host_ip"

  def test_reservation_enviroment_not_exists_get_server_ip_return_actual_host_ip(self):
    tfso_server = Server(5)
    assert tfso_server.get_server_ip() == util.get_ip_address()

  def test_reservation_enviroment_exists_start_listening_socket_return_socket_listening_to_environment_port_value(self):
    tfso_server = Server(1)
    with mock.patch.dict(os.environ, {'TFOS_SERVER_PORT': '9999'}):
      assert tfso_server.start_listening_socket().getsockname()[1] == 9999

  def test_reservation_enviroment_not_exists_start_listening_socket_return_socket(self):
    tfso_server = Server(1)
    print(tfso_server.start_listening_socket().getsockname()[1])
    assert type(tfso_server.start_listening_socket().getsockname()[1]) == int

  def test_reservation_server_multi(self):
    """Test reservation server, expecting multiple reservations"""
    num_clients = 4
    s = Server(num_clients)
    addr = s.start()

    def reserve(num):
      c = Client(addr)
      # time.sleep(random.randint(0,5))     # simulate varying start times
      resp = c.register({'node': num})
      self.assertEqual(resp, 'OK')
      c.await_reservations()
      c.close()

    # start/register clients
    threads = [None] * num_clients
    for i in range(num_clients):
      threads[i] = threading.Thread(target=reserve, args=(i,))
      threads[i].start()

    # wait for clients to complete
    for i in range(num_clients):
      threads[i].join()
    print("all done")

    # get list of reservations
    c = Client(addr)
    reservations = c.get_reservations()
    self.assertEqual(len(reservations), num_clients)

    # request server stop
    c.request_stop()
    time.sleep(1)
    self.assertEqual(s.done, True)


if __name__ == '__main__':
  unittest.main()
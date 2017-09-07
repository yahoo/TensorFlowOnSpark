import threading
import unittest

from tensorflowonspark.reservation import Reservations, Server, Client

class ReservationTest(unittest.TestCase):
  def test_reservation_class(self):
    """Test core reservation class, expecting 2 reservations"""
    r = Reservations(2)
    self.assertFalse(r.done())

    # add first reservation
    r.add({'node':1})
    self.assertFalse(r.done())
    self.assertEquals(r.remaining(), 1)

    # add second reservation
    r.add({'node':2})
    self.assertTrue(r.done())
    self.assertEquals(r.remaining(), 0)

    # get final list
    reservations = r.get()
    self.assertEquals(len(reservations), 2)

  def test_reservation_server(self):
    """Test reservation server, expecting 1 reservation"""
    s = Server(1)
    addr = s.start()

    # add first reservation
    c = Client(addr)
    resp = c.register({'node':1})
    self.assertEqual(resp, 'OK')

    # get list of reservations
    reservations = c.get_reservations()
    self.assertEquals(len(reservations), 1)

    # should return immediately with list of reservations
    reservations = c.await_reservations()
    self.assertEquals(len(reservations), 1)

    # request server stop
    c.request_stop()
    self.assertEquals(s.done, True)

  def test_reservation_server_multi(self):
    """Test reservation server, expecting multiple reservations"""
    num_clients = 4
    s = Server(num_clients)
    addr = s.start()

    def reserve(num):
      c = Client(addr)
      #time.sleep(random.randint(0,5))     # simulate varying start times
      resp = c.register({'node':num})
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
    self.assertEquals(len(reservations), num_clients)

    # request server stop
    c.request_stop()
    self.assertEquals(s.done, True)


if __name__ == '__main__':
  unittest.main()

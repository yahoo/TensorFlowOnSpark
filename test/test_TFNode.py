import getpass
import os
import unittest
from tensorflowonspark import TFManager, TFNode

class TFNodeTest(unittest.TestCase):
  def test_hdfs_path(self):
    """Normalization of absolution & relative string paths depending on filesystem"""
    cwd = os.getcwd()
    user = getpass.getuser()
    fs = ["file://", "hdfs://", "viewfs://"]
    paths = {
      "hdfs://foo/bar": ["hdfs://foo/bar", "hdfs://foo/bar", "hdfs://foo/bar"],
      "viewfs://foo/bar": ["viewfs://foo/bar", "viewfs://foo/bar", "viewfs://foo/bar"],
      "file://foo/bar": ["file://foo/bar", "file://foo/bar", "file://foo/bar"],
      "/foo/bar": ["file:///foo/bar", "hdfs:///foo/bar", "viewfs:///foo/bar"],
      "foo/bar": ["file://{}/foo/bar".format(cwd), "hdfs:///user/{}/foo/bar".format(user), "viewfs:///user/{}/foo/bar".format(user)],
    }

    for i in range(len(fs)):
      ctx = type('MockContext', (), {'defaultFS': fs[i], 'working_dir': cwd})
      for path, expected in paths.items():
        final_path = TFNode.hdfs_path(ctx, path)
        self.assertEqual(final_path, expected[i], "fs({}) + path({}) => {}, expected {}".format(fs[i], path, final_path, expected[i]))

  def test_datafeed(self):
    """TFNode.DataFeed basic operations"""
    mgr = TFManager.start('abc', ['input', 'output'], 'local')

    # insert 10 numbers followed by an end-of-feed marker
    q = mgr.get_queue('input')
    for i in range(10):
      q.put(i)
    q.put(None)                           # end-of-feed marker

    feed = TFNode.DataFeed(mgr)

    # [0,1]
    self.assertFalse(feed.done_feeding)
    batch = feed.next_batch(2)
    self.assertEqual(len(batch), 2)
    self.assertEqual(sum(batch), 1)

    # [2,3,4,5]
    self.assertFalse(feed.done_feeding)
    batch = feed.next_batch(4)
    self.assertEqual(len(batch), 4)
    self.assertEqual(sum(batch), 14)

    # [6,7,8,9]
    self.assertFalse(feed.done_feeding)
    batch = feed.next_batch(10)           # ask for more than available
    self.assertEqual(len(batch), 4)
    self.assertEqual(sum(batch), 30)

    # should be done
    self.assertTrue(feed.should_stop())


if __name__ == '__main__':
  unittest.main()

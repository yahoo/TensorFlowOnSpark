import unittest

from pyspark import SparkContext
from pyspark.sql import SparkSession

class SparkTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.sc = SparkContext('local[*]', cls.__name__)
    cls.spark = SparkSession.builder.getOrCreate()

  @classmethod
  def tearDownClass(cls):
    cls.spark.stop()
    cls.sc.stop()


class SimpleTest(SparkTest):
  def test_spark(self):
    sum = self.sc.parallelize(range(1000)).sum()
    self.assertEqual(499500, sum)


if __name__ == '__main__':
  unittest.main()

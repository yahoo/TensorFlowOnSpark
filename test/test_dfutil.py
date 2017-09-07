import os
import random
import shutil
import test
import unittest

from tensorflowonspark import dfutil

class DFUtilTest(test.SparkTest):
  @classmethod
  def setUpClass(cls):
    super(DFUtilTest, cls).setUpClass()

    # define model_dir and export_dir for tests
    cls.tfrecord_dir = os.getcwd() + os.sep + "test_tfr"

  @classmethod
  def tearDownClass(cls):
    super(DFUtilTest, cls).tearDownClass()

  def setUp(self):
    # remove any prior test artifacts
    shutil.rmtree(self.tfrecord_dir, ignore_errors=True)

  def tearDown(self):
    # Note: don't clean up artifacts after test (in case we need to view/debug)
    pass

  def test_dfutils(self):
    # create a DataFrame of a single row consisting of standard types (str, int, int_array, float, float_array)
    row1 = ('string_label', 1, [2, 3, 4, 5], -1.1, [-2.2, -3.3, -4.4, -5.5])
    rdd = self.sc.parallelize([row1])
    df1 = self.spark.createDataFrame(rdd, ['a', 'b', 'c', 'd', 'e'])

    # save the DataFrame as TFRecords
    dfutil.saveAsTFRecords(df1, self.tfrecord_dir)
    self.assertTrue(os.path.isdir(self.tfrecord_dir))

    # reload the DataFrame from exported TFRecords
    df2 = dfutil.loadTFRecords(self.sc, self.tfrecord_dir)
    row2 = df2.take(1)[0]

    print("row_saved: {}".format(row1))
    print("row_loaded: {}".format(row2))

    # confirm loaded values match original/saved values
    # note: strings/scalars are persisted as single-item lists
    self.assertEqual(row1[0], row2['a'][0])
    self.assertEqual(row1[1], row2['b'][0])
    self.assertEqual(row1[2], row2['c'])
    self.assertAlmostEqual(row1[3], row2['d'][0], 6)
    for i in range(len(row1[4])):
      self.assertAlmostEqual(row1[4][i], row2['e'][i], 6)

    # check origin of each DataFrame
    self.assertFalse(dfutil.isLoadedDF(df1))
    self.assertTrue(dfutil.isLoadedDF(df2))

    # references are equivalent
    df_ref = df2
    self.assertTrue(dfutil.isLoadedDF(df_ref))

    # mutated DFs are not equal, even if contents are identical
    df3 = df2.filter(df2.a[0] == 'string_label')
    self.assertFalse(dfutil.isLoadedDF(df3))

    # re-used/re-assigned variables are not equal
    df2 = df3
    self.assertFalse(dfutil.isLoadedDF(df2))


if __name__ == '__main__':
  unittest.main()

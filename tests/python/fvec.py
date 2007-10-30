import unittest

from aubio.aubiowrapper import *

buf_size = 2048
channels = 3

class fvec_test_case(unittest.TestCase):

  def setUp(self):
    self.vector = new_fvec(buf_size, channels)

  def tearDown(self):
    del_fvec(self.vector)

  def test_fvec(self):
    """ create and delete fvec """
    pass

  def test_fvec_read_sample(self):
    """ check new fvec elements are set to 0. """
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(fvec_read_sample(self.vector,channel,index),0.)

  def test_fvec_write_sample(self):
    """ check new fvec elements are set with fvec_write_sample """
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(self.vector,1.,channel,index)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(fvec_read_sample(self.vector,channel,index),1.)

if __name__ == '__main__':
  unittest.main()

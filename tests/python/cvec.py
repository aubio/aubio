import unittest

from aubio.aubiowrapper import *

buf_size = 2048
channels = 3

class cvec_test_case(unittest.TestCase):

  def setUp(self):
    self.vector = new_cvec(buf_size, channels)

  def tearDown(self):
    del_cvec(self.vector)

  def test_cvec(self):
    """ create and delete cvec """
    pass

  def test_cvec_read_norm(self):
    """ check new cvec norm elements are set to 0. """
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(cvec_read_norm(self.vector,channel,index),0.)

  def test_cvec_read_phas(self):
    """ check new cvec phas elements are set to 0. """
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(cvec_read_phas(self.vector,channel,index),0.)

  def test_cvec_write_norm(self):
    """ check new cvec norm elements are set with cvec_write_norm """
    for index in range(buf_size/2+1):
      for channel in range(channels):
        cvec_write_norm(self.vector,1.,channel,index)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(cvec_read_norm(self.vector,channel,index),1.)

  def test_cvec_write_phas(self):
    """ check new cvec phas elements are set with cvec_write_phas """
    for index in range(buf_size/2+1):
      for channel in range(channels):
        cvec_write_phas(self.vector,1.,channel,index)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(cvec_read_phas(self.vector,channel,index),1.)

if __name__ == '__main__':
  unittest.main()

import unittest

from aubio.aubiowrapper import *


buf_size = 1024
hop_size = 256
channels = 2

class aubio_phasevoc_test(unittest.TestCase):

  def setUp(self):
    self.o = new_aubio_pvoc(buf_size, hop_size, channels)

  def tearDown(self):
    del_aubio_pvoc(self.o)

  def test_create(self):
    """ test creation and deletion of phasevoc object """
    pass

  def test_zeroes(self):
    """ test phasevoc object on zeroes """
    input = new_fvec(hop_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    aubio_pvoc_do (self.o, input, fftgrain)
    aubio_pvoc_rdo(self.o, fftgrain, input)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(input, channel, index))

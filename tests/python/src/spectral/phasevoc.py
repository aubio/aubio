import unittest

from aubio.aubiowrapper import *

buf_size = 1024
hop_size = 256
channels = 2

precision = 6

class phasevoc_unit(unittest.TestCase):

  def setUp(self):
    self.o = new_aubio_pvoc(buf_size, hop_size, channels)

  def tearDown(self):
    del_aubio_pvoc(self.o)

  def test_create(self):
    """ create and delete phasevoc object """
    pass

  def test_zeroes(self):
    """ run phasevoc object on zeroes """
    input    = new_fvec(hop_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    output   = new_fvec(hop_size, channels)
    for index in range(hop_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(input, channel, index))
    aubio_pvoc_do (self.o, input, fftgrain)
    aubio_pvoc_rdo(self.o, fftgrain, output)
    for index in range(hop_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(output, channel, index))
    del input
    del fftgrain

  def test_ones(self):
    """ run phasevoc object on ones """
    input    = new_fvec(hop_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    output   = new_fvec(hop_size, channels)
    for index in range(hop_size):
      for channel in range(channels):
        fvec_write_sample(input, 1., channel, index)
        self.assertEqual(1., fvec_read_sample(input, channel, index))
    # make sure the first buf_size-hop_size samples are zeroes
    for i in range(buf_size/hop_size - 1):
      aubio_pvoc_do (self.o, input, fftgrain)
      aubio_pvoc_rdo(self.o, fftgrain, output)
      for index in range(hop_size):
        for channel in range(channels):
          self.assertAlmostEqual(0., fvec_read_sample(output, channel, index), precision)
    # make sure the first non zero input is correctly resynthesised
    aubio_pvoc_do (self.o, input, fftgrain)
    aubio_pvoc_rdo(self.o, fftgrain, output)
    for index in range(hop_size):
      for channel in range(channels):
        self.assertAlmostEqual(1., fvec_read_sample(output, channel, index), precision)
    del input
    del fftgrain

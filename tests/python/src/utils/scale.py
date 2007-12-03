import unittest

from template import aubio_unit_template
from aubio.aubiowrapper import *

buf_size = 2000
channels = 2

ilow = .40
ihig = 40.00
olow = 100.0
ohig = 1000.

class scale_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_scale(ilow, ihig, olow, ohig)

  def tearDown(self):
    del_aubio_scale(self.o)

  def test(self):
    """ create and delete scale """
    pass

  def test_zeroes(self):
    """ test scale on zeroes """
    input = new_fvec(buf_size, channels)
    aubio_scale_do(self.o, input)
    for index in range(buf_size):
      for channel in range(channels):
        expval = (- ilow) * (ohig - olow) / \
          (ihig - ilow) + olow
        val = fvec_read_sample(input, channel, index)
        self.assertCloseEnough(expval, val)
    del_fvec(input)

  def test_ilow(self):
    """ test scale on ilow """
    input = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(input, ilow, channel, index)
    aubio_scale_do(self.o, input)
    for index in range(buf_size):
      for channel in range(channels):
        val = fvec_read_sample(input, channel, index)
        self.assertAlmostEqual(olow, val)
    del_fvec(input)

  def test_ihig(self):
    """ test scale on ihig """
    input = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(input, ihig, channel, index)
    aubio_scale_do(self.o, input)
    for index in range(buf_size):
      for channel in range(channels):
        val = fvec_read_sample(input, channel, index)
        self.assertCloseEnough(ohig, val)
    del_fvec(input)

  def test_climbing_ramp(self):
    """ test scale on climbing ramp """
    input = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        rampval = index*(ihig-ilow)/buf_size + ilow 
        fvec_write_sample(input, rampval, channel, index)
    aubio_scale_do(self.o, input)
    for index in range(buf_size):
      for channel in range(channels):
        expval = index*(ohig-olow)/buf_size + olow
        self.assertCloseEnough(expval, \
          fvec_read_sample(input, channel, index))
    del_fvec(input)

  def test_falling_ramp(self):
    """ test scale on falling ramp """
    input = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(input, ihig \
          - index*(ihig-ilow)/buf_size, \
          channel, index)
    aubio_scale_do(self.o, input)
    for index in range(buf_size):
      for channel in range(channels):
        expval = ohig - index*(ohig-olow)/buf_size
        self.assertCloseEnough(expval, \
          fvec_read_sample(input, channel, index))
    del_fvec(input)

if __name__ == '__main__': unittest.main()

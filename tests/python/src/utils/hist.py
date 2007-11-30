import unittest

from template import aubio_unit_template
from aubio.aubiowrapper import *

buf_size = 2048
channels = 1
flow = 0.
fhig = 100.

nelems = 1000

class hist_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_hist(flow, fhig, nelems, channels)

  def tearDown(self):
    del_aubio_hist(self.o)

  def test_hist(self):
    """ create and delete hist """
    pass

  def test_hist_zeroes(self):
    """ test hist on zeroes """
    input = new_fvec(buf_size, channels)
    aubio_hist_do_notnull(self.o, input)
    aubio_hist_weight(self.o)
    self.assertEqual(0., aubio_hist_mean(self.o))
    del_fvec(input)

  def test_hist_impulse(self):
    """ test hist on impulse """
    input = new_fvec(buf_size, channels)
    constant = fhig - 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(1./nelems, aubio_hist_mean(self.o))
    del_fvec(input)

  def test_hist_impulse2(self):
    """ test hist on impulse """
    input = new_fvec(buf_size, channels)
    constant = fhig + 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(1./nelems, aubio_hist_mean(self.o))
    del_fvec(input)

if __name__ == '__main__': unittest.main()

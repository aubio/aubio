import unittest
import random

from template import aubio_unit_template
from aubio.aubiowrapper import *

buf_size = 2048
channels = 1
flow = float(random.randint(0, 100) + random.random())
fhig = float(random.randint(100, 1000) + random.random())

nelems = 1000

class hist_unit(aubio_unit_template):

  def setUp(self):
    print flow, fhig
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

  def test_hist_impulse_top(self):
    """ test hist on impulse (top - 1.) """
    """ this returns 1./nelems because 1 element is in the range """
    input = new_fvec(buf_size, channels)
    constant = fhig - 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(1./nelems, aubio_hist_mean(self.o))
    del_fvec(input)

  def test_hist_impulse_over(self):
    """ test hist on impulse (top + 1.) """
    """ this returns 0 because constant is out of range """
    input = new_fvec(buf_size, channels)
    constant = fhig + 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(0., aubio_hist_mean(self.o))
    del_fvec(input)

  def test_hist_impulse_bottom(self):
    """ test hist on constant near lower limit """
    """ this returns 1./nelems because 1 element is in the range """
    input = new_fvec(buf_size, channels)
    constant = flow + 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(1./nelems, aubio_hist_mean(self.o))
    del_fvec(input)

  def test_hist_impulse_under(self):
    """ test hist on constant under lower limit """
    """ this returns 0 because constant is out of range """
    input = new_fvec(buf_size, channels)
    constant = flow - 1. 
    fvec_write_sample(input,constant,0,0)
    aubio_hist_do_notnull(self.o, input)
    self.assertCloseEnough(0., aubio_hist_mean(self.o))
    del_fvec(input)

if __name__ == '__main__': unittest.main()

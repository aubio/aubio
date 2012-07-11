#! /usr/bin/python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from numpy import random
from aubio import cvec, filterbank

class aubio_filterbank_test_case(TestCase):

  def test_members(self):
    f = filterbank(40, 512)
    assert_equal ([f.n_filters, f.win_s], [40, 512])

  def test_set_coeffs(self):
    f = filterbank(40, 512)
    r = random.random([40, 512 / 2 + 1]).astype('float32')
    f.set_coeffs(r)
    assert_equal (r, f.get_coeffs())

if __name__ == '__main__':
  from unittest import main
  main()


#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from numpy import random
from math import pi
from numpy import array
from aubio import cvec, filterbank
from utils import array_from_text_file

class aubio_filterbank_test_case(TestCase):

  def test_members(self):
    f = filterbank(40, 512)
    assert_equal ([f.n_filters, f.win_s], [40, 512])

  def test_set_coeffs(self):
    f = filterbank(40, 512)
    r = random.random([40, 512 / 2 + 1]).astype('float32')
    f.set_coeffs(r)
    assert_equal (r, f.get_coeffs())

  def test_phase(self):
    f = filterbank(40, 512)
    c = cvec(512)
    c.phas[:] = pi
    assert_equal( f(c), 0);

  def test_norm(self):
    f = filterbank(40, 512)
    c = cvec(512)
    c.norm[:] = 1
    assert_equal( f(c), 0);

  def test_random_norm(self):
    f = filterbank(40, 512)
    c = cvec(512)
    c.norm[:] = random.random((512 / 2 + 1,)).astype('float32')
    assert_equal( f(c), 0)

  def test_random_coeffs(self):
    f = filterbank(40, 512)
    c = cvec(512)
    r = random.random([40, 512 / 2 + 1]).astype('float32')
    r /= r.sum()
    f.set_coeffs(r)
    c.norm[:] = random.random((512 / 2 + 1,)).astype('float32')
    assert_equal ( f(c) < 1., True )
    assert_equal ( f(c) > 0., True )

  def test_mfcc_coeffs(self):
    f = filterbank(40, 512)
    c = cvec(512)
    f.set_mel_coeffs_slaney(44100)
    c.norm[:] = random.random((512 / 2 + 1,)).astype('float32')
    assert_equal ( f(c) < 1., True )
    assert_equal ( f(c) > 0., True )

  def test_mfcc_coeffs_16000(self):
    expected = array_from_text_file('filterbank_mfcc_16000_512.expected')
    f = filterbank(40, 512)
    f.set_mel_coeffs_slaney(16000)
    assert_almost_equal ( expected, f.get_coeffs() )

if __name__ == '__main__':
  from unittest import main
  main()


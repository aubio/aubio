#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from numpy import array, shape
from aubio import cvec, filterbank

class aubio_filterbank_mel_test_case(TestCase):

  def test_slaney(self):
    f = filterbank(40, 512)
    f.set_mel_coeffs_slaney(16000)
    a = f.get_coeffs()
    assert_equal(shape (a), (40, 512/2 + 1) )

  def test_other_slaney(self):
    f = filterbank(40, 512*2)
    f.set_mel_coeffs_slaney(44100)
    a = f.get_coeffs()
    #print "sum is", sum(sum(a))
    for win_s in [256, 512, 1024, 2048, 4096]:
      f = filterbank(40, win_s)
      f.set_mel_coeffs_slaney(320000)
      a = f.get_coeffs()
      #print "sum is", sum(sum(a))

  def test_triangle_freqs_zeros(self):
    f = filterbank(9, 1024)
    freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
    freqs = array(freq_list, dtype = 'float32')
    f.set_triangle_bands(freqs, 48000)
    f.get_coeffs().T
    assert_equal ( f(cvec(1024)), 0)

  def test_triangle_freqs_ones(self):
    f = filterbank(9, 1024)
    freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
    freqs = array(freq_list, dtype = 'float32')
    f.set_triangle_bands(freqs, 48000)
    f.get_coeffs().T
    spec = cvec(1024)
    spec.norm[:] = 1
    assert_almost_equal ( f(spec),
            [ 0.02070313,  0.02138672,  0.02127604,  0.02135417, 
        0.02133301, 0.02133301,  0.02133311,  0.02133334,  0.02133345])

if __name__ == '__main__':
  from unittest import main
  main()



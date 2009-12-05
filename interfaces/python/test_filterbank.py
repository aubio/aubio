from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from numpy import array, shape
from _aubio import *

class aubio_filter_test_case(TestCase):

  def test_slaney(self):
    f = filterbank(40, 512)
    f.set_mel_coeffs_slaney(16000)
    a = f.get_coeffs()
    a.T

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

  def test_triangle_freqs(self):
    f = filterbank(9, 1024)
    freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
    freqs = array(freq_list, dtype = 'float32')
    f.set_triangle_bands(freqs, 48000)
    f.get_coeffs().T
    assert_equal ( f(cvec(1024)), [0] * 9)
    spec = cvec(1024)
    spec[0][40:100] = 100
    #print f(spec)

if __name__ == '__main__':
  from unittest import main
  main()


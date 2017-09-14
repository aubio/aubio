#! /usr/bin/env python

import numpy as np
from numpy.testing import TestCase, assert_equal, assert_almost_equal

from aubio import cvec, filterbank, float_type

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

class aubio_filterbank_mel_test_case(TestCase):

    def test_slaney(self):
        f = filterbank(40, 512)
        f.set_mel_coeffs_slaney(16000)
        a = f.get_coeffs()
        assert_equal(np.shape (a), (40, 512/2 + 1) )

    def test_other_slaney(self):
        f = filterbank(40, 512*2)
        f.set_mel_coeffs_slaney(44100)
        self.assertIsInstance(f.get_coeffs(), np.ndarray)
        #print "sum is", sum(sum(a))
        for win_s in [256, 512, 1024, 2048, 4096]:
            f = filterbank(40, win_s)
            f.set_mel_coeffs_slaney(32000)
            #print "sum is", sum(sum(a))
            self.assertIsInstance(f.get_coeffs(), np.ndarray)

    def test_triangle_freqs_zeros(self):
        f = filterbank(9, 1024)
        freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
        freqs = np.array(freq_list, dtype = float_type)
        f.set_triangle_bands(freqs, 48000)
        assert_equal ( f(cvec(1024)), 0)
        self.assertIsInstance(f.get_coeffs(), np.ndarray)

    def test_triangle_freqs_ones(self):
        f = filterbank(9, 1024)
        freq_list = [40, 80, 200, 400, 800, 1600, 3200, 6400, 12800, 15000, 24000]
        freqs = np.array(freq_list, dtype = float_type)
        f.set_triangle_bands(freqs, 48000)
        self.assertIsInstance(f.get_coeffs(), np.ndarray)
        spec = cvec(1024)
        spec.norm[:] = 1
        assert_almost_equal ( f(spec),
                [ 0.02070313, 0.02138672, 0.02127604, 0.02135417,
                    0.02133301, 0.02133301, 0.02133311, 0.02133334, 0.02133345])

if __name__ == '__main__':
    import nose2
    nose2.main()

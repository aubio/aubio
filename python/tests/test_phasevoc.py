#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, cvec, pvoc, float_type
from numpy import array, shape
from numpy.random import random
import numpy as np

precision = 4

class aubio_pvoc_test_case(TestCase):
    """ pvoc object test case """

    def test_members_automatic_sizes_default(self):
        """ check object creation with default parameters """
        f = pvoc()
        assert_equal ([f.win_s, f.hop_s], [1024, 512])

    def test_members_unnamed_params(self):
        """ check object creation with unnamed parameters """
        f = pvoc(2048, 128)
        assert_equal ([f.win_s, f.hop_s], [2048, 128])

    def test_members_named_params(self):
        """ check object creation with named parameters """
        f = pvoc(hop_s = 128, win_s = 2048)
        assert_equal ([f.win_s, f.hop_s], [2048, 128])

    def test_zeros(self):
        """ check the resynthesis of zeros gives zeros """
        win_s, hop_s = 1024, 256
        f = pvoc (win_s, hop_s)
        t = fvec (hop_s)
        for time in range( int ( 4 * win_s / hop_s ) ):
            s = f(t)
            r = f.rdo(s)
            assert_equal ( array(t), 0)
            assert_equal ( s.norm, 0)
            assert_equal ( s.phas, 0)
            assert_equal ( r, 0)

    def test_resynth_8_steps_sine(self):
        """ check the resynthesis of is correct with 87.5% overlap """
        hop_s = 1024
        ratio = 8
        freq = 445; samplerate = 22050
        sigin = np.sin( 2. * np.pi * np.arange(hop_s).astype(float_type) * freq / samplerate)
        r2 = self.reconstruction( sigin, hop_s, ratio)
        error = ((r2 - sigin)**2).mean()
        self.assertLessEqual(error , 1.e-3)

    def test_resynth_8_steps(self):
        """ check the resynthesis of is correct with 87.5% overlap """
        hop_s = 1024
        ratio = 8
        sigin = np.random.rand(hop_s).astype(float_type) * 2. - 1.
        r2 = self.reconstruction( sigin, hop_s, ratio)
        error = ((r2 - sigin)**2).mean()
        self.assertLessEqual(error , 1.e-2)

    def test_resynth_4_steps(self):
        """ check the resynthesis of is correct with 75% overlap """
        hop_s = 1024
        ratio = 4
        sigin = np.random.rand(hop_s).astype(float_type) * 2. - 1.
        r2 = self.reconstruction( sigin, hop_s, ratio)
        error = ((r2 - sigin)**2).mean()
        self.assertLessEqual(error , 1.e-2)

    def reconstruction(self, sigin, hop_s, ratio):
        buf_s = hop_s * ratio
        f = pvoc(buf_s, hop_s)
        zeros = fvec(hop_s)
        r2 = f.rdo( f(sigin) )
        for i in range(1, ratio):
            r2 = f.rdo( f(zeros) )
        return r2

    def plot_this( self, this ):
        from pylab import semilogy, show
        semilogy ( this )
        show ()

if __name__ == '__main__':
  from unittest import main
  main()


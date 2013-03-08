#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from aubio import fvec, fft, cvec
from numpy import array, shape
from math import pi

class aubio_fft_test_case(TestCase):

    def test_members(self):
        """ check members are set correctly """
        win_s = 2048
        f = fft(win_s)
        assert_equal (f.win_s, win_s)

    def test_output_dimensions(self):
        """ check the dimensions of output """
        win_s = 1024
        timegrain = fvec(win_s)
        f = fft (win_s)
        fftgrain = f (timegrain)
        assert_equal (shape(fftgrain.norm), (win_s/2+1,))
        assert_equal (shape(fftgrain.phas), (win_s/2+1,))

    def test_zeros(self):
        """ check the transform of zeros is all zeros """
        win_s = 512
        timegrain = fvec(win_s)
        f = fft (win_s)
        fftgrain = f (timegrain)
        assert_equal ( fftgrain.norm, 0 )
        assert_equal ( fftgrain.phas, 0 )

    def test_impulse(self):
        """ check the transform of one impulse at a random place """
        from random import random
        from math import floor
        win_s = 256
        i = floor(random()*win_s)
        impulse = pi * random()
        f = fft(win_s)
        timegrain = fvec(win_s)
        timegrain[i] = impulse
        fftgrain = f ( timegrain )
        #self.plot_this ( fftgrain.phas )
        assert_almost_equal ( fftgrain.norm, impulse, decimal = 6 )
        assert_equal ( fftgrain.phas <= pi, True)
        assert_equal ( fftgrain.phas >= -pi, True)

    def test_impulse_negative(self):
        """ check the transform of one impulse at a random place """
        from random import random
        from math import floor
        win_s = 256
        i = 0
        impulse = -10.
        f = fft(win_s)
        timegrain = fvec(win_s)
        timegrain[i] = impulse
        fftgrain = f ( timegrain )
        #self.plot_this ( fftgrain.phas )
        assert_almost_equal ( fftgrain.norm, abs(impulse), decimal = 6 )
        if impulse < 0:
            # phase can be pi or -pi, as it is not unwrapped
            assert_almost_equal ( abs(fftgrain.phas[1:-1]) , pi, decimal = 6 )
            assert_almost_equal ( fftgrain.phas[0], pi, decimal = 6)
            assert_almost_equal ( fftgrain.phas[-1], pi, decimal = 6)
        else:
            assert_equal ( fftgrain.phas[1:-1] == 0, True)
            assert_equal ( fftgrain.phas[0] == 0, True)
            assert_equal ( fftgrain.phas[-1] == 0, True)
        # now check the resynthesis
        synthgrain = f.rdo ( fftgrain )
        #self.plot_this ( fftgrain.phas.T )
        assert_equal ( fftgrain.phas <= pi, True)
        assert_equal ( fftgrain.phas >= -pi, True)
        #self.plot_this ( synthgrain - timegrain )
        assert_almost_equal ( synthgrain, timegrain, decimal = 6 )

    def test_impulse_at_zero(self):
        """ check the transform of one impulse at a index 0 """
        win_s = 1024
        impulse = pi
        f = fft(win_s)
        timegrain = fvec(win_s)
        timegrain[0] = impulse
        fftgrain = f ( timegrain )
        #self.plot_this ( fftgrain.phas )
        assert_equal ( fftgrain.phas[0], 0)
        # could be 0 or -0 depending on fft implementation (0 for fftw3, -0 for ooura)
        assert_almost_equal ( fftgrain.phas[1], 0)
        assert_almost_equal ( fftgrain.norm[0], impulse, decimal = 6 )

    def test_rdo_before_do(self):
        """ check running fft.rdo before fft.do works """
        win_s = 1024
        impulse = pi
        f = fft(win_s)
        fftgrain = cvec(win_s)
        t = f.rdo( fftgrain )
        assert_equal ( t, 0 )

    def plot_this(self, this):
        from pylab import plot, show
        plot ( this )
        show ()

if __name__ == '__main__':
    from unittest import main
    main()


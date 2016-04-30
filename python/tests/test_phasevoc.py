#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_array_less
from aubio import fvec, cvec, pvoc, float_type
from numpy import array, shape
from numpy.random import random
from nose2.tools import params
import numpy as np

if float_type == 'float32':
    max_sq_error = 1.e-12
else:
    max_sq_error = 1.e-29

def create_sine(hop_s, freq, samplerate):
    t = np.arange(hop_s).astype(float_type)
    return np.sin( 2. * np.pi * freq * t / float(samplerate))

def create_noise(hop_s):
    return np.random.rand(hop_s).astype(float_type) * 2. - 1.

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
            assert_equal ( t, 0.)
            assert_equal ( s.norm, 0.)
            assert_equal ( s.phas, 0.)
            assert_equal ( r, 0.)

    def test_resynth_8_steps_sine(self):
        """ check the resynthesis of is correct with 87.5% overlap """
        hop_s = 1024
        ratio = 8
        freq = 445; samplerate = 22050
        sigin = create_sine(hop_s, freq, samplerate)
        self.reconstruction( sigin, hop_s, ratio)

    def test_resynth_8_steps(self):
        """ check the resynthesis of is correct with 87.5% overlap """
        hop_s = 1024
        ratio = 8
        sigin = create_noise(hop_s)
        self.reconstruction(sigin, hop_s, ratio)

    def test_resynth_4_steps_sine(self):
        """ check the resynthesis of is correct with 87.5% overlap """
        hop_s = 1024
        ratio = 4
        freq = 445; samplerate = 22050
        sigin = create_sine(hop_s, freq, samplerate)
        self.reconstruction(sigin, hop_s, ratio)

    def test_resynth_4_steps(self):
        """ check the resynthesis of is correct with 75% overlap """
        hop_s = 1024
        ratio = 4
        sigin = create_noise(hop_s)
        self.reconstruction(sigin, hop_s, ratio)

    def test_resynth_2_steps_sine(self):
        """ check the resynthesis of is correct with 50% overlap """
        hop_s = 1024
        ratio = 2
        freq = 445; samplerate = 22050
        sigin = create_sine(hop_s, freq, samplerate)
        self.reconstruction(sigin, hop_s, ratio)

    def test_resynth_2_steps(self):
        """ check the resynthesis of is correct with 50% overlap """
        hop_s = 1024
        ratio = 2
        sigin = create_noise(hop_s)
        self.reconstruction(sigin, hop_s, ratio)

    @params(
            ( 256, 8),
            ( 256, 4),
            ( 256, 2),
            ( 512, 8),
            ( 512, 4),
            ( 512, 2),
            (1024, 8),
            (1024, 4),
            (1024, 2),
            (2048, 8),
            (2048, 4),
            (2048, 2),
            (4096, 8),
            (4096, 4),
            (4096, 2),
            (8192, 8),
            (8192, 4),
            (8192, 2),
            )
    def test_resynth_steps_noise(self, hop_s, ratio):
        """ check the resynthesis of a random signal is correct """
        sigin = create_noise(hop_s)
        self.reconstruction(sigin, hop_s, ratio)

    @params(
            (44100,  256, 8,   441),
            (44100,  256, 4,  1203),
            (44100,  256, 2,  3045),
            (44100,  512, 8,   445),
            (44100,  512, 4,   445),
            (44100,  512, 2,   445),
            (44100, 1024, 8,   445),
            (44100, 1024, 4,   445),
            (44100, 1024, 2,   445),
            ( 8000, 1024, 2,   445),
            (22050, 1024, 2,   445),
            (22050,  256, 8,   445),
            (96000, 1024, 8, 47000),
            (96000, 1024, 8,    20),
            )
    def test_resynth_steps_sine(self, samplerate, hop_s, ratio, freq):
        """ check the resynthesis of a sine is correct """
        sigin = create_sine(hop_s, freq, samplerate)
        self.reconstruction(sigin, hop_s, ratio)

    def reconstruction(self, sigin, hop_s, ratio):
        buf_s = hop_s * ratio
        f = pvoc(buf_s, hop_s)
        zeros = fvec(hop_s)
        r2 = f.rdo( f(sigin) )
        for i in range(1, ratio):
            r2 = f.rdo( f(zeros) )
        # compute square errors
        sq_error = (r2 - sigin)**2
        # make sure all square errors are less than desired precision
        assert_array_less(sq_error, max_sq_error)

if __name__ == '__main__':
    from nose2 import main
    main()


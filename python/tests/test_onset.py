#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
# WARNING: numpy also has an fft object
from aubio import onset, cvec
from numpy import array, shape, arange, zeros, log
from math import pi

class aubio_onset(TestCase):

    def test_members(self):
        o = onset()
        assert_equal ([o.buf_size, o.hop_size, o.method, o.samplerate],
            [1024,512,'default',44100])
    

if __name__ == '__main__':
    from unittest import main
    main()

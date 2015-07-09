#! /usr/bin/env python

from numpy.testing import TestCase
from numpy.testing.utils import assert_almost_equal
from aubio import window

class aubio_window(TestCase):

    def test_accept_name_and_size(self):
        window("default", 1024)

    def test_fail_name_not_string(self):
        try:
            window(10, 1024)
        except ValueError, e:
            pass
        else:
            self.fail('non-string window type does not raise a ValueError')

    def test_fail_size_not_int(self):
        try:
            window("default", "default")
        except ValueError, e:
            pass
        else:
            self.fail('non-integer window length does not raise a ValueError')

    def test_compute_hanning_1024(self):
        from numpy import cos, arange
        from math import pi
        size = 1024
        aubio_window = window("hanning", size)
        numpy_window = .5 - .5 * cos(2. * pi * arange(size) / size)
        assert_almost_equal(aubio_window, numpy_window)

if __name__ == '__main__':
    from unittest import main
    main()

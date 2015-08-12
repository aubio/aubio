#! /usr/bin/env python

from numpy.testing import TestCase
from numpy.testing.utils import assert_equal, assert_almost_equal
from numpy import cos, arange
from math import pi

from aubio import window, level_lin, db_spl, silence_detection, level_detection

from aubio import fvec

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
        size = 1024
        aubio_window = window("hanning", size)
        numpy_window = .5 - .5 * cos(2. * pi * arange(size) / size)
        assert_almost_equal(aubio_window, numpy_window)

class aubio_level_lin(TestCase):
    def test_accept_fvec(self):
        level_lin(fvec(1024))

    def test_fail_not_fvec(self):
        try:
            level_lin("default")
        except ValueError, e:
            pass
        else:
            self.fail('non-number input phase does not raise a TypeError')

    def test_zeros_is_zeros(self):
        assert_equal(level_lin(fvec(1024)), 0.)

    def test_minus_ones_is_one(self):
        from numpy import ones
        assert_equal(level_lin(-ones(1024, dtype="float32")), 1.)

class aubio_db_spl(TestCase):
    def test_accept_fvec(self):
        db_spl(fvec(1024))

    def test_fail_not_fvec(self):
        try:
            db_spl("default")
        except ValueError, e:
            pass
        else:
            self.fail('non-number input phase does not raise a TypeError')

    def test_zeros_is_inf(self):
        from math import isinf
        assert isinf(db_spl(fvec(1024)))

    def test_minus_ones_is_zero(self):
        from numpy import ones
        assert_equal(db_spl(-ones(1024, dtype="float32")), 0.)

class aubio_silence_detection(TestCase):
    def test_accept_fvec(self):
        silence_detection(fvec(1024), -70.)

    def test_fail_not_fvec(self):
        try:
            silence_detection("default", -70)
        except ValueError, e:
            pass
        else:
            self.fail('non-number input phase does not raise a TypeError')

    def test_zeros_is_one(self):
        from math import isinf
        assert silence_detection(fvec(1024), -70) == 1

    def test_minus_ones_is_zero(self):
        from numpy import ones
        assert silence_detection(ones(1024, dtype="float32"), -70) == 0

class aubio_level_detection(TestCase):
    def test_accept_fvec(self):
        level_detection(fvec(1024), -70.)

    def test_fail_not_fvec(self):
        try:
            level_detection("default", -70)
        except ValueError, e:
            pass
        else:
            self.fail('non-number input phase does not raise a TypeError')

    def test_zeros_is_one(self):
        from math import isinf
        assert level_detection(fvec(1024), -70) == 1

    def test_minus_ones_is_zero(self):
        from numpy import ones
        assert level_detection(ones(1024, dtype="float32"), -70) == 0

if __name__ == '__main__':
    from unittest import main
    main()

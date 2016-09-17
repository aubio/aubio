#! /usr/bin/env python

from nose2 import TestCase
import aubio

class aubio_pitchshift(TestCase):

    def setUp(self):
        self.o = aubio.pitchshift()

    def test_default_creation(self):
        self.assertEqual(self.o.get_pitchscale(), 1)
        self.assertEqual(self.o.get_transpose(), 0)

    def test_on_zeros(self):
        test_length = 20000
        read = 0
        # test on zeros
        vec = aubio.fvec(512)
        transpose_range = 24
        while read < test_length:
            # transpose the samples
            out = self.o(vec)
            self.assertTrue((out == 0).all())
            # position in the file (between 0. and 1.)
            percent_read = read / float(test_length)
            # variable transpose rate (in semitones)
            transpose = 2 * transpose_range * percent_read - transpose_range
            # set transpose rate
            self.o.set_transpose(transpose)
            read += len(vec)

    def test_transpose_too_high(self):
        with self.assertRaises(ValueError):
            self.o.set_transpose(24.3)

    def test_transpose_too_low(self):
        with self.assertRaises(ValueError):
            self.o.set_transpose(-24.3)

if __name__ == '__main__':
    from nose2 import main
    main()

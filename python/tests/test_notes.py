#! /usr/bin/env python

from unittest import main
from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import notes

AUBIO_DEFAULT_NOTES_SILENCE = -70.
AUBIO_DEFAULT_NOTES_MINIOI_MS = 30.

class aubio_notes_default(TestCase):

    def test_members(self):
        o = notes()
        assert_equal ([o.buf_size, o.hop_size, o.method, o.samplerate],
            [1024,512,'default',44100])


class aubio_notes_params(TestCase):

    samplerate = 44100

    def setUp(self):
        self.o = notes(samplerate = self.samplerate)

    def test_get_minioi_ms(self):
        assert_equal (self.o.get_minioi_ms(), AUBIO_DEFAULT_NOTES_MINIOI_MS)

    def test_set_minioi_ms(self):
        val = 40.
        self.o.set_minioi_ms(val)
        assert_almost_equal (self.o.get_minioi_ms(), val)

    def test_get_silence(self):
        assert_equal (self.o.get_silence(), AUBIO_DEFAULT_NOTES_SILENCE)

    def test_set_silence(self):
        val = -50
        self.o.set_silence(val)
        assert_equal (self.o.get_silence(), val)

if __name__ == '__main__':
    main()

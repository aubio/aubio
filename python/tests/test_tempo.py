#! /usr/bin/env python

from unittest import main
from numpy.testing import TestCase, assert_equal, assert_almost_equal
import aubio
import numpy as np
from aubio import float_type, tempo

class aubio_tempo_default(TestCase):

    def test_members(self):
        o = aubio.tempo()
        assert_equal ([o.buf_size, o.hop_size, o.method, o.samplerate],
            [1024,512,'default',44100])

class aubio_tempo_params(TestCase):

    samplerate = 44100

    def setUp(self):
        self.o = aubio.tempo(samplerate = self.samplerate)

    def test_get_delay(self):
        self.assertEqual(self.o.get_delay(), 0)

    def test_set_delay(self):
        val = 256
        self.o.set_delay(val)
        assert_equal (self.o.get_delay(), val)

    def test_set_negative_delay(self):
        """ a negative delay comes back negative, not as 2**32 - n (aubio#151) """
        self.o.set_delay(-100)
        assert_equal (self.o.get_delay(), -100)

    def test_get_delay_s(self):
        self.assertEqual(self.o.get_delay_s(), 0.)

    def test_set_delay_s(self):
        val = .05
        self.o.set_delay_s(val)
        assert_almost_equal (self.o.get_delay_s(), val)

    def test_get_delay_ms(self):
        self.assertEqual(self.o.get_delay_ms(), 0.)

    def test_set_delay_ms(self):
        val = 50.
        self.o.set_delay_ms(val)
        assert_almost_equal (self.o.get_delay_ms(), val)

    def test_get_threshold(self):
        assert_almost_equal(self.o.get_threshold(), 0.3)

    def test_set_threshold(self):
        val = .1
        self.o.set_threshold(val)
        assert_almost_equal (self.o.get_threshold(), val)

    def test_get_silence(self):
        self.assertEqual(self.o.get_silence(), -90.)

    def test_set_silence(self):
        val = -50.
        self.o.set_silence(val)
        assert_almost_equal (self.o.get_silence(), val)

    def test_get_last(self):
        self.assertEqual(self.o.get_last(), 0.)

    def test_get_last_s(self):
        self.assertEqual(self.o.get_last_s(), 0.)

    def test_get_last_ms(self):
        self.assertEqual(self.o.get_last_ms(), 0.)

    def test_get_period(self):
        self.assertEqual(self.o.get_period(), 0.)

    def test_get_period_s(self):
        self.assertEqual(self.o.get_period_s(), 0.)

    def test_get_last_tatum(self):
        self.assertEqual(self.o.get_last_tatum(), 0.)

    def test_set_tatum_signature(self):
        self.o.set_tatum_signature(8)
        self.o.set_tatum_signature(64)
        self.o.set_tatum_signature(1)

    def test_set_wrong_tatum_signature(self):
        with self.assertRaises(ValueError):
            self.o.set_tatum_signature(101)
        with self.assertRaises(ValueError):
            self.o.set_tatum_signature(0)

def click_train(bpm, samplerate, seconds=20):
    """ short noise bursts every beat """
    n = int(seconds * samplerate)
    signal = np.zeros(n, dtype=float_type)
    rng = np.random.default_rng(0)
    burst_len = int(0.02 * samplerate)
    envelope = np.exp(-np.arange(burst_len) / (0.004 * samplerate))
    for start in np.arange(0, n - burst_len, 60. * samplerate / bpm):
        start = int(round(start))
        signal[start:start + burst_len] += rng.standard_normal(burst_len) * envelope
    return signal


class aubio_tempo_samplerates(TestCase):
    """ the 206 bpm ceiling holds whatever the detection frame rate:
    it was a fixed 25 frames, 144 bpm at 30000/500 (LedFx) and 103 bpm at
    22050/512, so faster tempos were halved (aubio#284) """

    def found_bpm(self, bpm, samplerate, hop_size, win_s):
        o = tempo("default", win_s, hop_size, samplerate)
        signal = click_train(bpm, samplerate)
        for i in range(0, len(signal) - hop_size, hop_size):
            o(signal[i:i + hop_size])
        return o.get_bpm()

    def assert_tracks(self, samplerate, hop_size, win_s):
        for bpm in (120., 150., 160.):
            found = self.found_bpm(bpm, samplerate, hop_size, win_s)
            assert abs(found - bpm) < .05 * bpm, (bpm, found)

    def test_ledfx_rate(self):
        self.assert_tracks(30000, 500, 4096)

    def test_22050(self):
        self.assert_tracks(22050, 512, 1024)

    def test_44100_unchanged(self):
        self.assert_tracks(44100, 512, 1024)

if __name__ == '__main__':
    main()

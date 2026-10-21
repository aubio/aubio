#! /usr/bin/env python

"""aubio reads numpy arrays in place, so the bindings must respect their
layout (aubio#311). Arrays aubio only reads may be strided, misaligned,
byte-swapped or read-only: they are copied when needed. Arrays aubio writes
into must be the caller's own contiguous, writeable buffer, or the call
fails instead of writing somewhere else."""

import numpy as np
from numpy.testing import TestCase, assert_equal
from aubio import (cvec, filterbank, float_type, min_removal, pitch, shift,
                   zero_crossing_rate)


def alternating(n):
    x = np.zeros(n, dtype=float_type)
    x[1::2] = 1.
    return x


class aubio_strided_inputs(TestCase):

    def test_every_other_sample(self):
        # x[::2] is all zeros: no crossings, not the alternating buffer
        x = alternating(1024)
        assert_equal(zero_crossing_rate(x[::2]), 0.)
        assert_equal(zero_crossing_rate(x[1::2]), 0.)

    def test_one_channel_of_interleaved_stereo(self):
        samplerate, hop_size = 44100, 512
        t = np.arange(hop_size * 20) / samplerate
        left = np.sin(2 * np.pi * 440. * t)
        right = np.sin(2 * np.pi * 1000. * t)
        stereo = np.stack([left, right], axis=1).astype(float_type)
        p = pitch('yin', 2048, hop_size, samplerate)
        found = [p(stereo[i:i + hop_size, 0])[0]
                 for i in range(0, len(stereo), hop_size)]
        assert abs(np.median(found[5:]) - 440.) < 5.

    def test_byte_swapped(self):
        x = alternating(512)
        swapped = x.astype(x.dtype.newbyteorder())
        assert_equal(zero_crossing_rate(swapped), zero_crossing_rate(x))

    def test_read_only(self):
        # np.frombuffer over bytes, as audio callbacks deliver it
        x = np.frombuffer(alternating(512).tobytes(), dtype=float_type)
        assert not x.flags.writeable
        assert_equal(zero_crossing_rate(x), zero_crossing_rate(alternating(512)))

    def test_strided_rows(self):
        f = filterbank(40, 512)
        wide = np.random.random((40, 514)).astype(float_type)
        f.set_coeffs(wide[:, ::2])
        assert_equal(f.get_coeffs(), wide[:, ::2])

    def test_one_dimensional_matrix(self):
        f = filterbank(40, 512)
        with self.assertRaises(ValueError):
            f.set_coeffs(np.zeros(257, dtype=float_type))

    def test_wrong_dtype_still_refused(self):
        # no silent float64 -> float32 conversion
        wrong = 'float64' if float_type == 'float32' else 'float32'
        with self.assertRaises(ValueError):
            zero_crossing_rate(np.zeros(512, dtype=wrong))


class aubio_in_place(TestCase):
    """ min_removal and shift write into their argument """

    def test_writes_into_the_array(self):
        x = np.arange(8, dtype=float_type) + 1.
        min_removal(x)
        assert_equal(x, np.arange(8, dtype=float_type))
        y = np.arange(8, dtype=float_type)
        shift(y)
        assert_equal(y, [4, 5, 6, 7, 0, 1, 2, 3])

    def test_read_only_refused(self):
        x = np.frombuffer(np.ones(8, dtype=float_type).tobytes(),
                          dtype=float_type)
        with self.assertRaises(ValueError):
            min_removal(x)
        with self.assertRaises(ValueError):
            shift(x)
        assert_equal(x, np.ones(8))

    def test_strided_refused(self):
        x = np.ones(16, dtype=float_type)
        with self.assertRaises(ValueError):
            min_removal(x[::2])
        assert_equal(x, np.ones(16))


class aubio_cvec_arrays(TestCase):

    def test_contiguous_array_is_shared(self):
        c = cvec(512)
        norm = np.zeros(257, dtype=float_type)
        c.norm = norm
        norm[3] = 1.
        assert_equal(c.norm[3], 1.)

    def test_read_only_or_strided_array_is_copied(self):
        c = cvec(512)
        ro = np.frombuffer(np.ones(257, dtype=float_type).tobytes(),
                           dtype=float_type)
        c.norm = ro
        assert c.norm.flags.writeable
        assert_equal(c.norm, ro)
        wide = np.arange(514, dtype=float_type)
        c.phas = wide[::2]
        assert c.phas.flags.c_contiguous
        assert_equal(c.phas, wide[::2])


if __name__ == '__main__':
    from unittest import main
    main()

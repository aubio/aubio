#! /usr/bin/env python

from numpy.testing import TestCase
from numpy.testing import assert_equal, assert_almost_equal
from numpy import random, sin, arange, mean, median
from math import pi
from aubio import fvec, pitch

class aubio_mathutils_test_case(TestCase):

  def test_members(self):
    p = pitch()
    assert_equal ( [p.method, p.buf_size, p.hop_size, p.samplerate],
      ['default', 1024, 512, 44100])

  def test_members_not_default(self):
    p = pitch('mcomb', 2048, 512, 32000)
    assert_equal ( [p.method, p.buf_size, p.hop_size, p.samplerate],
      ['mcomb', 2048, 512, 32000])

  def test_run_on_zeros(self):
    p = pitch('mcomb', 2048, 512, 32000)
    f = fvec (512)
    assert_equal ( p(f), 0. )

  def test_run_on_ones(self):
    p = pitch('mcomb', 2048, 512, 32000)
    f = fvec (512)
    f[:] = 1
    assert( p(f) != 0. )

  def test_run_default_on_sinusoid(self):
    method = 'default'
    buf_size = 2048
    hop_size = 512
    samplerate = 32000
    freq = 450.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def test_run_schmitt_on_sinusoid(self):
    method = 'schmitt'
    buf_size = 4096
    hop_size = 512
    samplerate = 44100
    freq = 800.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def test_run_mcomb_on_sinusoid(self):
    method = 'mcomb'
    buf_size = 2048
    hop_size = 512
    samplerate = 44100
    freq = 10000.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def test_run_fcomb_on_sinusoid(self):
    method = 'fcomb'
    buf_size = 2048
    hop_size = 512
    samplerate = 32000
    freq = 440.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def test_run_yin_on_sinusoid(self):
    method = 'yin'
    buf_size = 4096
    hop_size = 512
    samplerate = 32000
    freq = 880.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def test_run_yinfft_on_sinusoid(self):
    method = 'yinfft'
    buf_size = 2048
    hop_size = 512
    samplerate = 32000
    freq = 640.
    self.run_pitch_on_sinusoid(method, buf_size, hop_size, samplerate, freq)

  def run_pitch_on_sinusoid(self, method, buf_size, hop_size, samplerate, freq):
    p = pitch(method, buf_size, hop_size, samplerate)
    sinvec = self.build_sinusoid(hop_size * 100, freq, samplerate)
    self.run_pitch(p, sinvec, freq)

  def build_sinusoid(self, length, freq, samplerate):
    return sin( 2. * pi * arange(length).astype('float32') * freq / samplerate)

  def run_pitch(self, p, input_vec, freq):
    count = 0
    pitches, errors = [], []
    for vec_slice in input_vec.reshape((-1, p.hop_size)):
      pitch = p(vec_slice)
      pitches.append(pitch)
      errors.append(1. - pitch / freq)
    # check that the mean of all relative errors is less than 10%
    assert_almost_equal (mean(errors), 0., decimal = 2)

if __name__ == '__main__':
  from unittest import main
  main()


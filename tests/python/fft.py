import unittest
import math

from aubio.aubiowrapper import *

buf_size = 8092 
channels = 4

precision = 6

class aubio_mfft_test_case(unittest.TestCase):

  def setUp(self):
    self.o = new_aubio_mfft(buf_size, channels)

  def tearDown(self):
    del_aubio_mfft(self.o)

  def test_create(self):
    """ test creation and deletion of fft object """
    pass

  def test_aubio_mfft_do_zeroes(self):
    """ test aubio_mfft_do on zeroes """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(input, channel, index))
    aubio_mfft_do(self.o, input, fftgrain)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(0., cvec_read_phas(fftgrain, channel, index))
    del fftgrain
    del input

  def test_aubio_mfft_rdo_zeroes(self):
    """ test aubio_mfft_rdo on zeroes """
    fftgrain = new_cvec(buf_size, channels)
    output    = new_fvec(buf_size, channels)
    aubio_mfft_rdo(self.o, fftgrain, output)
    # check output
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(output, channel, index))
    del fftgrain
    del output

  def test_aubio_mfft_do_impulse(self):
    """ test aubio_mfft_do with an impulse on one channel """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # write impulse in channel 0, sample 0.
    some_constant = 0.3412432456
    fvec_write_sample(input, some_constant, 0, 0)
    aubio_mfft_do(self.o, input, fftgrain)
    # check norm
    for index in range(buf_size/2+1):
      self.assertAlmostEqual(some_constant, cvec_read_norm(fftgrain, 0, index), precision)
    for index in range(buf_size/2+1):
      for channel in range(1, channels):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    # check phas
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(0., cvec_read_phas(fftgrain, channel, index))
    del fftgrain
    del input

  def test_aubio_mfft_do_constant(self):
    """ test aubio_mfft_do with a constant on one channel """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # write impulse in channel 0, sample 0.
    some_constant = 0.003412432456
    for index in range(1,buf_size):
      fvec_write_sample(input, some_constant, 0, index)
    aubio_mfft_do(self.o, input, fftgrain)
    # check norm and phase == 0 in all other channels 
    for index in range(buf_size/2+1):
      for channel in range(1, channels):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    # check norm and phase == 0 in first first and last bin of first channel
    self.assertAlmostEqual((buf_size-1)*some_constant, cvec_read_norm(fftgrain, 0, 0), precision)
    self.assertEqual(0., cvec_read_phas(fftgrain, 0, 0))
    self.assertEqual(0., cvec_read_norm(fftgrain, 0, buf_size/2+1))
    self.assertEqual(0., cvec_read_phas(fftgrain, 0, buf_size/2+1))
    # check unwrap2pi(phas) ~= pi everywhere but in first bin
    for index in range(1,buf_size/2+1):
       self.assertAlmostEqual ( math.pi, aubio_unwrap2pi(cvec_read_phas(fftgrain, 0, index)), precision)
       self.assertAlmostEqual(some_constant, cvec_read_norm(fftgrain, 0, index), precision)
    del fftgrain
    del input

  def test_aubio_mfft_do_impulse_multichannel(self):
    " test aubio_mfft_do on impulse two channels "
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # put an impulse in first an last channel, at first and last index
    fvec_write_sample(input, 1., 0, 0)
    fvec_write_sample(input, 1., channels-1, 0)
    aubio_mfft_do(self.o, input, fftgrain)
    # check the norm
    for index in range(buf_size/2+1):
      self.assertEqual(1., cvec_read_norm(fftgrain, 0, index))
    for index in range(buf_size/2+1):
      for channel in range(1, channels-1):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    for index in range(buf_size/2+1):
      self.assertEqual(1., cvec_read_norm(fftgrain, channels-1, index))
    # check the phase
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(0., cvec_read_phas(fftgrain, channel, index))
    del fftgrain
    del input

  def test_aubio_mfft_rdo_impulse(self):
    """ test aubio_mfft_rdo on impulse """
    fftgrain  = new_cvec(buf_size, channels)
    for channel in range(channels):
      cvec_write_norm(fftgrain, 1., channel, 0)
    output    = new_fvec(buf_size, channels)
    aubio_mfft_rdo(self.o, fftgrain, output)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertAlmostEqual(fvec_read_sample(output, channel, index), 1./buf_size, precision)
    del fftgrain
    del output

  def test_aubio_mfft_do_back_and_forth(self):
    """ test aubio_mfft_rdo on a constant """
    input    = new_fvec(buf_size, channels)
    output   = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        fvec_write_sample(input, 0.67, channel, index)
    aubio_mfft_do(self.o, input, fftgrain)
    aubio_mfft_rdo(self.o, fftgrain, output)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertAlmostEqual(fvec_read_sample(output, channel, index), 0.67, precision)
    del fftgrain
    del output

if __name__ == '__main__': unittest.main()

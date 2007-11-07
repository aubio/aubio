import math

from template import aubio_unit_template

from aubio.aubiowrapper import *

buf_size = 1024
channels = 4

class fft_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_fft(buf_size, channels)

  def tearDown(self):
    del_aubio_fft(self.o)

  def test_create(self):
    """ test creation and deletion of fft object """
    pass

  def test_do_zeroes(self):
    """ test aubio_fft_do on zeroes """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertCloseEnough(0., fvec_read_sample(input, channel, index))
    aubio_fft_do(self.o, input, fftgrain)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertCloseEnough(0., cvec_read_norm(fftgrain, channel, index))
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertCloseEnough(0., cvec_read_phas(fftgrain, channel, index))
    del fftgrain
    del input

  def test_rdo_zeroes(self):
    """ test aubio_fft_rdo on zeroes """
    fftgrain = new_cvec(buf_size, channels)
    output    = new_fvec(buf_size, channels)
    aubio_fft_rdo(self.o, fftgrain, output)
    # check output
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(output, channel, index))
    del fftgrain
    del output

  def test_do_impulse(self):
    """ test aubio_fft_do with an impulse on one channel """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # write impulse in channel 0, sample 0.
    some_constant = 0.3412432456
    fvec_write_sample(input, some_constant, 0, 0)
    aubio_fft_do(self.o, input, fftgrain)
    # check norm
    for index in range(buf_size/2+1):
      self.assertCloseEnough(some_constant, cvec_read_norm(fftgrain, 0, index))
    for index in range(buf_size/2+1):
      for channel in range(1, channels):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    # check phas
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(0., cvec_read_phas(fftgrain, channel, index))
    del fftgrain
    del input

  def test_do_constant(self):
    """ test aubio_fft_do with a constant on one channel """
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # write impulse in channel 0, sample 0.
    some_constant = 0.003412432456
    for index in range(1,buf_size):
      fvec_write_sample(input, some_constant, 0, index)
    aubio_fft_do(self.o, input, fftgrain)
    # check norm and phase == 0 in all other channels 
    for index in range(buf_size/2+1):
      for channel in range(1, channels):
        self.assertEqual(0., cvec_read_norm(fftgrain, channel, index))
    # check norm and phase == 0 in first first and last bin of first channel
    self.assertCloseEnough((buf_size-1)*some_constant, cvec_read_norm(fftgrain, 0, 0))
    self.assertCloseEnough(0., cvec_read_phas(fftgrain, 0, 0))
    self.assertCloseEnough(0., cvec_read_norm(fftgrain, 0, buf_size/2+1))
    self.assertCloseEnough(0., cvec_read_phas(fftgrain, 0, buf_size/2+1))
    # check unwrap2pi(phas) ~= pi everywhere but in first bin
    for index in range(1,buf_size/2+1):
       self.assertCloseEnough(math.pi, aubio_unwrap2pi(cvec_read_phas(fftgrain, 0, index)))
       self.assertCloseEnough(some_constant, cvec_read_norm(fftgrain, 0, index))
    del fftgrain
    del input

  def test_do_impulse_multichannel(self):
    " test aubio_fft_do on impulse two channels "
    input    = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    # put an impulse in first an last channel, at first and last index
    fvec_write_sample(input, 1., 0, 0)
    fvec_write_sample(input, 1., channels-1, 0)
    aubio_fft_do(self.o, input, fftgrain)
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

  def test_rdo_impulse(self):
    """ test aubio_fft_rdo on impulse """
    fftgrain  = new_cvec(buf_size, channels)
    for channel in range(channels):
      cvec_write_norm(fftgrain, 1., channel, 0)
    output    = new_fvec(buf_size, channels)
    aubio_fft_rdo(self.o, fftgrain, output)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertCloseEnough(fvec_read_sample(output, channel, index), 1./buf_size)
    del fftgrain
    del output

  def test_do_back_and_forth(self):
    """ test aubio_fft_rdo on a constant """
    input    = new_fvec(buf_size, channels)
    output   = new_fvec(buf_size, channels)
    fftgrain = new_cvec(buf_size, channels)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        fvec_write_sample(input, 0.67, channel, index)
    aubio_fft_do(self.o, input, fftgrain)
    aubio_fft_rdo(self.o, fftgrain, output)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertCloseEnough(0.67, fvec_read_sample(output, channel, index))
    del fftgrain
    del output

if __name__ == '__main__': unittest.main()

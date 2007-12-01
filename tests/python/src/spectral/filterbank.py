
import unittest
from aubio.aubiowrapper import * 

win_size = 2048 
channels = 1
n_filters = 40
samplerate = 44100
zerodb = -96.015602111816406

class filterbank_test_case(unittest.TestCase):
  
  def setUp(self):
      self.input_spectrum = new_cvec(win_size,channels)
      self.output_banks = new_fvec(n_filters,channels)
      self.filterbank = new_aubio_filterbank(n_filters,win_size)

  def tearDown(self):
      del_aubio_filterbank(self.filterbank)
      del_cvec(self.input_spectrum)
      del_fvec(self.output_banks)

  def testzeroes(self):
      """ check the output of the filterbank is -96 when input spectrum is 0 """
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

  def testphase(self):
      """ check the output of the filterbank is -96 when input phase is pi """
      from math import pi
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_phas(self.input_spectrum,pi,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

  def testones(self):
      """ check the output of the filterbank is -96 when input norm is 1
          (the filterbank is currently set to 0).
      """
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_norm(self.input_spectrum,1.,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

  def testmfcc_zeroes(self):
      """ check the mfcc filterbank output is -96 when input is 0 """
      self.filterbank = new_aubio_filterbank_mfcc(n_filters, win_size, samplerate, 0., samplerate) 
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

  def testmfcc_phasepi(self):
      """ check the mfcc filterbank output is -96 when input phase is pi """
      self.filterbank = new_aubio_filterbank_mfcc(n_filters, win_size, samplerate, 0., samplerate) 
      from math import pi
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_phas(self.input_spectrum,pi,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

  def testmfcc_ones(self):
      """ check setting the input spectrum to 1 gives something between -3. and -4. """ 
      self.filterbank = new_aubio_filterbank_mfcc(n_filters, win_size, samplerate, 0., samplerate) 
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_norm(self.input_spectrum,1.,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          val = fvec_read_sample(self.output_banks,channel,index)
          self.failIf(val > -2.5 , val )
          self.failIf(val < -4. , val )

  def testmfcc_channels(self):
      """ check the values of each filters in the mfcc filterbank """
      import os.path
      self.filterbank = new_aubio_filterbank_mfcc(n_filters, win_size, samplerate, 
        0., samplerate) 
      filterbank_mfcc = [ [float(f) for f in line.strip().split()]
        for line in open(os.path.join('src','spectral','filterbank_mfcc.txt')).readlines()]
      for channel in range(n_filters):
        vec = aubio_filterbank_getchannel(self.filterbank,channel)
        for index in range(win_size): 
          self.assertAlmostEqual(fvec_read_sample(vec,0,index),
            filterbank_mfcc[channel][index])
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), zerodb)

if __name__ == '__main__':
    unittest.main()

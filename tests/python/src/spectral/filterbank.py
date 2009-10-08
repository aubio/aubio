from template import aubio_unit_template
from localaubio import *

win_size = 2048 
channels = 1
n_filters = 40
samplerate = 44100

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
      """ check the output of the filterbank is 0 when input spectrum is 0 """
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

  def testphase(self):
      """ check the output of the filterbank is 0 when input phase is pi """
      from math import pi
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_phas(self.input_spectrum,pi,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

  def testones(self):
      """ check the output of the filterbank is 0 when input norm is 1
          (the filterbank is currently set to 0).
      """
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_norm(self.input_spectrum,1.,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

  def testmfcc_zeroes(self):
      """ check the mfcc filterbank output is 0 when input is 0 """
      self.filterbank = new_aubio_filterbank(n_filters, win_size) 
      aubio_filterbank_do(self.filterbank, self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

  def testmfcc_phasepi(self):
      """ check the mfcc filterbank output is 0 when input phase is pi """
      self.filterbank = new_aubio_filterbank(n_filters, win_size) 
      aubio_filterbank_set_mel_coeffs_slaney(self.filterbank, samplerate)
      from math import pi
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_phas(self.input_spectrum,pi,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

  def testmfcc_ones(self):
      """ check setting the input spectrum to 1 gives something between 0. and 1. """ 
      self.filterbank = new_aubio_filterbank(n_filters, win_size) 
      aubio_filterbank_set_mel_coeffs_slaney(self.filterbank, samplerate)
      for channel in range(channels):
        for index in range(win_size/2+1): 
          cvec_write_norm(self.input_spectrum,1.,channel,index)
      aubio_filterbank_do(self.filterbank,self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          val = fvec_read_sample(self.output_banks,channel,index)
          self.failIf(val < 0. , val )
          self.failIf(val > 1. , val )

  def testmfcc_channels(self):
      """ check the values of each filters in the mfcc filterbank """
      import os.path
      win_size = 512
      self.filterbank = new_aubio_filterbank(n_filters, win_size) 
      aubio_filterbank_set_mel_coeffs_slaney(self.filterbank, 16000)
      filterbank_mfcc = array_from_text_file ( 
          os.path.join('src','spectral','filterbank_mfcc_16000_512.txt') )
      vec = aubio_filterbank_get_coeffs(self.filterbank)
      for channel in range(n_filters):
        for index in range(win_size/2+1): 
          self.assertAlmostEqual(fvec_read_sample(vec,channel,index),
            filterbank_mfcc[channel][index])
      aubio_filterbank_do(self.filterbank, self.input_spectrum,
        self.output_banks) 
      for channel in range(channels):
        for index in range(n_filters): 
          self.assertEqual(fvec_read_sample(self.output_banks,channel,index), 0)

if __name__ == '__main__':
    unittest.main()

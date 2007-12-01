from template import aubio_unit_template
from aubio.aubiowrapper import *

samplerate = 44100
buf_size = 1024
channels = 2

class adsgn_filter_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_adsgn_filter(samplerate, channels)

  def tearDown(self):
    del_aubio_adsgn_filter(self.o)

  def test_creation(self):
    pass

  def test_filter_zeroes(self):
    """ check filter run on a vector full of zeroes returns zeros """
    vec = new_fvec(buf_size, channels)
    aubio_adsgn_filter_do(self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

  def test_filter_ones(self):
    vec = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(vec, 1., channel, index)
    aubio_adsgn_filter_do(self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertNotEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

  def test_filter_denormal(self):
    vec = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(vec, 1.e-37, channel, index)
    aubio_adsgn_filter_do(self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

if __name__ == '__main__':
  unittest.main()

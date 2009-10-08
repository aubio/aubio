from template import aubio_unit_template
from localaubio import *

samplerate = 44100
buf_size = 1024
channels = 2

class filter_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_filter(samplerate, 8, channels)

  def tearDown(self):
    del_aubio_filter(self.o)

  def test_creation(self):
    """ check filter creation and deletion """
    pass

  def test_filter_zeroes(self):
    """ check filter run on a vector full of zeroes returns zeros """
    vec = new_fvec(buf_size, channels)
    aubio_filter_do(self.o, vec)
    for index in range(buf_size/2+1):
      for channel in range(channels):
        self.assertEqual(fvec_read_sample(vec,channel,index),0.)
    del_fvec(vec)

if __name__ == '__main__':
  unittest.main()

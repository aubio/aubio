from localaubio import *

samplerate = 44100
buf_size = 1024
channels = 1

class a_weighting_unit (aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_filter_a_weighting (samplerate, channels)

  def tearDown(self):
    del_aubio_filter (self.o)

  def test_creation(self):
    pass

  def test_filter_zeroes(self):
    """ check filter run on a vector full of zeroes returns zeros """
    vec = new_fvec(buf_size, channels)
    aubio_filter_do (self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

  def test_filter_ones(self):
    vec = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(vec, 1., channel, index)
    aubio_filter_do (self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertNotEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

  def test_filter_denormal(self):
    vec = new_fvec(buf_size, channels)
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(vec, 2.e-42, channel, index)
    aubio_filter_do (self.o, vec)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(vec,channel,index))
    del_fvec(vec)

  def test_simple(self):
    buf_size = 32
    input = new_fvec (buf_size, 1)
    output = new_fvec (buf_size, 1)
    expected = array_from_text_file('src/temporal/a_weighting_test_simple.expected')
    fvec_write_sample (input, 0.5, 0, 12)
    for i in range(buf_size):
      for c in range(channels):
        self.assertEqual(expected[0][i], fvec_read_sample(input, c, i))
    f = new_aubio_filter_a_weighting (samplerate, channels)
    aubio_filter_do_outplace (f, input, output)
    del_aubio_filter (f)
    for i in range(buf_size):
      for c in range(channels):
        self.assertCloseEnough(expected[1][i], fvec_read_sample(output, c, i))

if __name__ == '__main__':
  import unittest
  unittest.main()

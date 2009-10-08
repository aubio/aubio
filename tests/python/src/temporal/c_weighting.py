from localaubio import *

class c_weighting_unit(aubio_unit_template):

  def test_simple(self):
    expected = array_from_text_file('src/temporal/c_weighting_test_simple.expected')
    samplerate = 44100
    channels = 1
    buf_size = 32

    # prepare input
    input = new_fvec (buf_size, 1)
    output = new_fvec (buf_size, 1)
    fvec_write_sample (input, 0.5, 0, 12)

    # check input
    for i in range(buf_size):
      for c in range(channels):
        self.assertEqual(expected[0][i], fvec_read_sample(input, c, i))

    # filter
    f = new_aubio_filter_c_weighting (samplerate, channels)
    aubio_filter_do_outplace (f, input, output)
    del_aubio_filter (f)

    # check output
    for i in range(buf_size):
      for c in range(channels):
        self.assertAlmostEqual(expected[1][i], fvec_read_sample(output, c, i))

  def test_simple_8000(self):
    expected = array_from_text_file('src/temporal/c_weighting_test_simple_8000.expected')
    samplerate = 8000 
    channels = 1
    buf_size = 32

    # prepare input
    input = new_fvec (buf_size, 1)
    output = new_fvec (buf_size, 1)
    fvec_write_sample (input, 0.5, 0, 12)

    # check input
    for i in range(buf_size):
      for c in range(channels):
        self.assertEqual(expected[0][i], fvec_read_sample(input, c, i))

    # filter
    f = new_aubio_filter_c_weighting (samplerate, channels)
    aubio_filter_do_outplace (f, input, output)
    del_aubio_filter (f)

    # check output
    for i in range(buf_size):
      for c in range(channels):
        self.assertAlmostEqual(expected[1][i], fvec_read_sample(output, c, i))

if __name__ == '__main__':
  import unittest
  unittest.main()

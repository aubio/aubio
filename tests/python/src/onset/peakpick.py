from template import aubio_unit_template
from localaubio import *

buf_size = 7 
channels = 1

class peakpick_unit(aubio_unit_template):

  def setUp(self):
    self.o = new_aubio_peakpicker(1)
    aubio_peakpicker_set_threshold (0.1)
    pass

  def tearDown(self):
    del_aubio_peakpicker(self.o)
    pass

  def test_peakpick(self):
    """ create and delete peakpick """
    pass

  def test_peakpick_zeroes(self):
    """ check peakpick run on a vector full of zero returns no peak. """
    self.assertEqual(0., aubio_peakpicker_get_thresholded_input(self.o))

  def test_peakpick_impulse(self):
    """ check peakpick detects a single impulse as a peak. """
    """ check two consecutive peaks are detected as one. """
    #print 
    for index in range(0,buf_size-1):
      input = new_fvec(buf_size, channels)
      fvec_write_sample(input, 1000., 0, index)
      fvec_write_sample(input, 1000./2, 0, index+1)
      #print "%2s" % index, aubio_peakpicker_do(self.o, input), "|",
      #for i in range(buf_size): print fvec_read_sample(input, 0, i),
      #print
      del_fvec(input)

  def test_peakpick_consecutive_peaks(self):
    """ check two consecutive peaks are detected as one. """
    #print 
    for index in range(0,buf_size-4):
      input = new_fvec(buf_size, channels)
      fvec_write_sample(input, 1000./2, 0, index)
      fvec_write_sample(input, 1000., 0, index+1)
      fvec_write_sample(input, 1000., 0, index+3)
      fvec_write_sample(input, 1000./2, 0, index+4)
      peak_pick_result = aubio_peakpicker_do(self.o, input)
      if index == 2:
        # we are at the peak. check the result is after current sample,
        # and not after next one
        self.failIf( 1. >= peak_pick_result )
        self.failIf( 2. < peak_pick_result )
      else: self.assertEqual(0., peak_pick_result)
      del_fvec(input)
    for index in range(buf_size-4,buf_size-1):
      input = new_fvec(buf_size, channels)
      fvec_write_sample(input, 1000./2, 0, index)
      fvec_write_sample(input, 1000., 0, index+1)
      peak_pick_result = aubio_peakpicker_do(self.o, input)
      self.assertEqual(0., peak_pick_result)
      del_fvec(input)

  def test_peakpick_set_threshold(self):
    """ test aubio_peakpicker_set_threshold """
    new_threshold = 0.1
    aubio_peakpicker_set_threshold(self.o, new_threshold)
    self.assertCloseEnough(new_threshold, aubio_peakpicker_get_threshold(self.o))

  def test_peakpick_get_threshold(self):
    """ test aubio_peakpicker_get_threshold """
    new_threshold = aubio_peakpicker_get_threshold(self.o) 
    aubio_peakpicker_set_threshold(self.o, new_threshold)
    self.assertCloseEnough(new_threshold, aubio_peakpicker_get_threshold(self.o))

if __name__ == '__main__':
  unittest.main()

#! /usr/bin/python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, source
from numpy import array

path = "/Users/piem/archives/sounds/loops/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav"

class aubio_filter_test_case(TestCase):

  def test_members(self):
    f = source(path)
    print dir(f)

  def test_read(self):
    f = source(path)
    total_frames = 0
    while True:
      vec, read = f()
      total_frames += read
      if read < f.hop_size: break
    print "read", total_frames / float(f.samplerate), " seconds from", path

if __name__ == '__main__':
  from unittest import main
  main()


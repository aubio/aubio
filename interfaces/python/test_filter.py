#! /usr/bin/python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, digital_filter
from numpy import array

def array_from_text_file(filename, dtype = 'float'):
  return array([line.split() for line in open(filename).readlines()], 
      dtype = dtype)

class aubio_filter_test_case(TestCase):

  def test_members(self):
    f = digital_filter()
    assert_equal (f.order, 7)
    f = digital_filter(5)
    assert_equal (f.order, 5)
    f(fvec())
  
  def test_cweighting_error(self):
    f = digital_filter (2)
    self.assertRaises ( ValueError, f.set_c_weighting, 44100 )
    f = digital_filter (8)
    self.assertRaises ( ValueError, f.set_c_weighting, 44100 )
    f = digital_filter (5)
    self.assertRaises ( ValueError, f.set_c_weighting, 4000 )
    f = digital_filter (5)
    self.assertRaises ( ValueError, f.set_c_weighting, 193000 )
    f = digital_filter (7)
    self.assertRaises ( ValueError, f.set_a_weighting, 193000 )
    f = digital_filter (5)
    self.assertRaises ( ValueError, f.set_a_weighting, 192000 )

  def test_c_weighting(self):
    expected = array_from_text_file('c_weighting_test_simple.expected')
    f = digital_filter(5)
    f.set_c_weighting(44100)
    v = fvec(32)
    v[12] = .5
    u = f(v)
    assert_almost_equal (expected[1], u)

  def test_a_weighting(self):
    expected = array_from_text_file('a_weighting_test_simple.expected')
    f = digital_filter(7)
    f.set_a_weighting(44100)
    v = fvec(32)
    v[12] = .5
    u = f(v)
    assert_almost_equal (expected[1], u)

  def test_a_weighting_parted(self):
    expected = array_from_text_file('a_weighting_test_simple.expected')
    f = digital_filter(7)
    f.set_a_weighting(44100)
    v = fvec(16)
    v[12] = .5
    u = f(v)
    assert_almost_equal (expected[1][:16], u)
    # one more time
    v = fvec(16)
    u = f(v)
    assert_almost_equal (expected[1][16:], u)

if __name__ == '__main__':
  from unittest import main
  main()


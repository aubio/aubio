from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal as numpy_assert_equal
from _aubio import *
from numpy import array

AUBIO_DO_CASTING = 0

def assert_equal(a, b):
  numpy_assert_equal(array(a),array(b))

class aubiomodule_test_case(TestCase):

  def setUp(self):
    """ try importing aubio """

  def test_vector(self):
    a = fvec()
    a.length, a.channels
    a[0]
    array(a)
    a = fvec(10)
    a = fvec(1, 2)
    array(a).T
    a[0] = range(a.length)
    a[1][0] = 2

  def test_wrong_values(self):
    self.assertRaises (ValueError, fvec, -10)
    self.assertRaises (ValueError, fvec, 1, -1)
  
    a = fvec(2, 3)
    self.assertRaises (IndexError, a.__getitem__, 3)
    self.assertRaises (IndexError, a[0].__getitem__, 2)

  def test_alpha_norm_of_fvec(self):
    a = fvec(2, 2)
    self.assertEquals (alpha_norm(a, 1), 0)
    a[0] = [1, 2] 
    self.assertEquals (alpha_norm(a, 1), 1.5)
    a[1] = [1, 2] 
    self.assertEquals (alpha_norm(a, 1), 3)
    a[0] = [0, 1]; a[1] = [1, 0]
    self.assertEquals (alpha_norm(a, 2), 1)

  def test_alpha_norm_of_array_of_float32(self):
    # check scalar fails
    a = array(1, dtype = 'float32')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    # check 3d array fails
    a = array([[[1,2],[3,4]]], dtype = 'float32')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    # check 1d array
    a = array(range(10), dtype = 'float32')
    self.assertEquals (alpha_norm(a, 1), 4.5)
    # check 2d array
    a = array([range(10), range(10)], dtype = 'float32')
    self.assertEquals (alpha_norm(a, 1), 9)

  def test_alpha_norm_of_array_of_int(self):
    a = array(1, dtype = 'int')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array([[[1,2],[3,4]]], dtype = 'int')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array(range(10), dtype = 'int')
    self.assertRaises (ValueError, alpha_norm, a, 1)

  def test_alpha_norm_of_array_of_string (self):
    a = "hello"
    self.assertRaises (ValueError, alpha_norm, a, 1)

  def test_zero_crossing_rate(self):
    a = array([0,1,-1], dtype='float32')
    self.assertEquals (zero_crossing_rate(a), 1./3 )
    a = array([0.]*100, dtype='float32')
    self.assertEquals (zero_crossing_rate(a), 0 )
    a = array([-1.]*100, dtype='float32')
    self.assertEquals (zero_crossing_rate(a), 0 )
    a = array([1.]*100, dtype='float32')
    self.assertEquals (zero_crossing_rate(a), 0 )

  def test_alpha_norm_of_array_of_float64(self):
    # check scalar fail
    a = array(1, dtype = 'float64')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    # check 3d array fail
    a = array([[[1,2],[3,4]]], dtype = 'float64')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    if AUBIO_DO_CASTING:
      # check float64 1d array fail
      a = array(range(10), dtype = 'float64')
      self.assertEquals (alpha_norm(a, 1), 4.5)
      # check float64 2d array fail
      a = array([range(10), range(10)], dtype = 'float64')
      self.assertEquals (alpha_norm(a, 1), 9)
    else:
      # check float64 1d array fail
      a = array(range(10), dtype = 'float64')
      self.assertRaises (ValueError, alpha_norm, a, 1)
      # check float64 2d array fail
      a = array([range(10), range(10)], dtype = 'float64')
      self.assertRaises (ValueError, alpha_norm, a, 1)

  def test_fvec_min_removal_of_array(self):
    a = array([20,1,19], dtype='float32')
    b = min_removal(a)
    assert_equal (array(b), [19, 0, 18])
    assert_equal (b, [19, 0, 18])
    assert_equal (a, b)
    a[0] = 0
    assert_equal (a, b)

  def test_fvec_min_removal_of_array_float64(self):
    a = array([20,1,19], dtype='float64')
    if AUBIO_DO_CASTING:
      b = min_removal(a)
      assert_equal (array(b), [19, 0, 18])
      assert_equal (b, [19, 0, 18])
      #assert_equal (a, b)
    else:
      self.assertRaises (ValueError, min_removal, a)
      
  def test_fvec_min_removal_of_fvec(self):
    a = fvec(3, 1)
    a[0] = [20, 1, 19]
    b = min_removal(a)
    assert_equal (array(b), [19, 0, 18])
    assert_equal (b, [19, 0, 18])
    assert_equal (a, b)

if __name__ == '__main__':
  from unittest import main
  main()


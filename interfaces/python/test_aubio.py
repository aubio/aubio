import unittest
from _aubio import *
from numpy import array

class aubiomodule_test_case(unittest.TestCase):

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
    a = array(1, dtype = 'float32')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array([[[1,2],[3,4]]], dtype = 'float32')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array(range(10), dtype = 'float32')
    self.assertEquals (alpha_norm(a, 1), 4.5)
    a = array([range(10), range(10)], dtype = 'float32')
    self.assertEquals (alpha_norm(a, 1), 9)

  def test_alpha_norm_of_array_of_float64(self):
    a = array(1, dtype = 'float64')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array([[[1,2],[3,4]]], dtype = 'float64')
    self.assertRaises (ValueError, alpha_norm, a, 1)
    a = array(range(10), dtype = 'float64')
    self.assertEquals (alpha_norm(a, 1), 4.5)
    a = array([range(10), range(10)], dtype = 'float64')
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

if __name__ == '__main__':
  unittest.main()

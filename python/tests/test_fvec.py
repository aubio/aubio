#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from aubio import fvec, zero_crossing_rate, alpha_norm, min_removal
from numpy import array, shape

default_size = 512

class aubio_fvec_test_case(TestCase):

    def test_vector_created_with_zeroes(self):
        a = fvec(10)
        assert a.dtype == 'float32'
        assert a.shape == (10,)
        assert_equal (a, 0)

    def test_vector_create_with_list(self):
        a = fvec([0,1,2,3])
        assert a.dtype == 'float32'
        assert a.shape == (4,)
        assert_equal (range(4), a)

    def test_vector_assign_element(self):
        a = fvec(default_size)
        a[0] = 1
        assert_equal(a[0], 1)

    def test_vector_assign_element_end(self):
        a = fvec(default_size)
        a[-1] = 1
        assert_equal(a[-1], 1)
        assert_equal(a[len(a)-1], 1)

    def test_vector(self):
        a = fvec()
        a, len(a) #a.length
        a[0]
        array(a)
        a = fvec(10)
        a = fvec(1)
        a.T
        array(a).T
        a = range(len(a))

    def test_wrong_values(self):
        self.assertRaises (ValueError, fvec, -10)
  
        a = fvec(2)
        self.assertRaises (IndexError, a.__getitem__, 3)
        self.assertRaises (IndexError, a.__getitem__, 2)

    def test_alpha_norm_of_fvec(self):
        a = fvec(2)
        self.assertEquals (alpha_norm(a, 1), 0)
        a[0] = 1
        self.assertEquals (alpha_norm(a, 1), 0.5)
        a[1] = 1
        self.assertEquals (alpha_norm(a, 1), 1)
        a = array([0, 1], dtype='float32')
        from math import sqrt
        assert_almost_equal (alpha_norm(a, 2), sqrt(2)/2.)

    def test_alpha_norm_of_none(self):
        self.assertRaises (ValueError, alpha_norm, None, 1)

    def test_alpha_norm_of_array_of_float32(self):
        # check scalar fails
        a = array(1, dtype = 'float32')
        self.assertRaises (ValueError, alpha_norm, a, 1)
        # check 2d array fails
        a = array([[2],[4]], dtype = 'float32')
        self.assertRaises (ValueError, alpha_norm, a, 1)
        # check 1d array
        a = array(range(10), dtype = 'float32')
        self.assertEquals (alpha_norm(a, 1), 4.5)

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
        assert_almost_equal (zero_crossing_rate(a), 1./3. )
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
        self.assertRaises (ValueError, min_removal, a)

    def test_fvec_min_removal_of_fvec(self):
        a = fvec(3)
        a = array([20, 1, 19], dtype = 'float32')
        b = min_removal(a)
        assert_equal (array(b), [19, 0, 18])
        assert_equal (b, [19, 0, 18])
        assert_equal (a, b)

if __name__ == '__main__':
    from unittest import main
    main()

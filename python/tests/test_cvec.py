#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from aubio import cvec
from numpy import array, shape, pi

class aubio_cvec_test_case(TestCase):

    def test_vector_created_with_zeroes(self):
        a = cvec(10)
        shape(a.norm)
        shape(a.phas)
        a.norm[0]
        assert_equal(a.norm, 0.)
        assert_equal(a.phas, 0.)

    def test_vector_assign_element(self):
        a = cvec()
        a.norm[0] = 1
        assert_equal(a.norm[0], 1)
        a.phas[0] = 1
        assert_equal(a.phas[0], 1)

    def test_vector_assign_element_end(self):
        a = cvec()
        a.norm[-1] = 1
        assert_equal(a.norm[-1], 1)
        assert_equal(a.norm[len(a.norm)-1], 1)
        a.phas[-1] = 1
        assert_equal(a.phas[-1], 1)
        assert_equal(a.phas[len(a.phas)-1], 1)

    def test_assign_cvec_norm_slice(self):
        spec = cvec(1024)
        spec.norm[40:100] = 100
        assert_equal(spec.norm[0:40], 0)
        assert_equal(spec.norm[40:100], 100)
        assert_equal(spec.norm[100:-1], 0)
        assert_equal(spec.phas, 0)

    def test_assign_cvec_phas_slice(self):
        spec = cvec(1024)
        spec.phas[39:-1] = -pi
        assert_equal(spec.phas[0:39], 0)
        assert_equal(spec.phas[39:-1], -pi)
        assert_equal(spec.norm, 0)

if __name__ == '__main__':
    from unittest import main
    main()

#! /usr/bin/env python

from nose2 import main
from nose2.tools import params
from numpy import random, count_nonzero
from numpy.testing import TestCase
from aubio import mfcc, cvec, float_type

buf_size = 2048
n_filters = 40
n_coeffs = 13
samplerate = 44100


new_params = ['buf_size', 'n_filters', 'n_coeffs', 'samplerate']
new_deflts = [1024, 40, 13, 44100]

class aubio_mfcc(TestCase):

    def setUp(self):
        self.o = mfcc()

    def test_default_creation(self):
        pass

    def test_delete(self):
        del self.o

    @params(*new_params)
    def test_read_only_member(self, name):
        o = self.o
        with self.assertRaises((TypeError, AttributeError)):
            setattr(o, name, 0)

    @params(*zip(new_params, new_deflts))
    def test_default_param(self, name, expected):
        """ test mfcc.{:s} = {:d} """.format(name, expected)
        o = self.o
        self.assertEqual( getattr(o, name), expected)

class aubio_mfcc_wrong_params(TestCase):

    def test_wrong_buf_size(self):
        with self.assertRaises(ValueError):
            mfcc(buf_size = -1)

    def test_wrong_n_filters(self):
        with self.assertRaises(ValueError):
            mfcc(n_filters = -1)

    def test_wrong_n_coeffs(self):
        with self.assertRaises(ValueError):
            mfcc(n_coeffs = -1)

    def test_wrong_samplerate(self):
        with self.assertRaises(ValueError):
            mfcc(samplerate = -1)

    def test_wrong_input_size(self):
        m = mfcc(buf_size = 1024)
        with self.assertRaises(ValueError):
            m(cvec(512))

class aubio_mfcc_compute(TestCase):

    def test_members(self):

        o = mfcc(buf_size, n_filters, n_coeffs, samplerate)
        #assert_equal ([o.buf_size, o.method], [buf_size, method])

        spec = cvec(buf_size)
        #spec.norm[0] = 1
        #spec.norm[1] = 1./2.
        #print "%20s" % method, str(o(spec))
        coeffs = o(spec)
        self.assertEqual(coeffs.size, n_coeffs)
        #print coeffs
        spec.norm = random.random_sample((len(spec.norm),)).astype(float_type)
        spec.phas = random.random_sample((len(spec.phas),)).astype(float_type)
        #print "%20s" % method, str(o(spec))
        self.assertEqual(count_nonzero(o(spec) != 0.), n_coeffs)
        #print coeffs


class aubio_mfcc_all_parameters(TestCase):

    @params(
            (2048, 40, 13, 44100),
            (1024, 40, 13, 44100),
            (512, 40, 13, 44100),
            (512, 40, 13, 16000),
            (256, 40, 13, 16000),
            (128, 40, 13, 16000),
            (128, 40, 12, 16000),
            (128, 40, 13, 15000),
            (512, 40, 20, 44100),
            (512, 40, 40, 44100),
            (512, 40, 3, 44100),
            (1024, 40, 20, 44100),
            #(1024, 30, 20, 44100),
            (1024, 40, 40, 44100),
            (1024, 40, 3, 44100),
            )
    def test_run_with_params(self, buf_size, n_filters, n_coeffs, samplerate):
        " check mfcc can run with reasonable parameters "
        o = mfcc(buf_size, n_filters, n_coeffs, samplerate)
        spec = cvec(buf_size)
        spec.phas[0] = 0.2
        for _ in range(10):
            o(spec)
        #print coeffs

if __name__ == '__main__':
    main()

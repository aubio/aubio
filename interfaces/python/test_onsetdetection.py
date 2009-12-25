from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
# WARNING: numpy also has an fft object
from aubio import specdesc, cvec
from numpy import array, shape, arange, zeros, log
from math import pi

class aubio_specdesc(TestCase):

    def test_members(self):
        o = specdesc()
        assert_equal ([o.buf_size, o.method],
            [1024, "default"])

    def test_hfc(self):
        o = specdesc("hfc")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        assert_equal (a, c.norm)
        assert_equal ( sum(a*(a+1)), o(c))

    def test_complex(self):
        o = specdesc("complex")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        assert_equal (a, c.norm)
        # the previous run was on zeros, so previous frames are still 0
        # so we have sqrt ( abs ( r2 ^ 2) ) == r2
        assert_equal ( sum(a), o(c))
        # second time. c.norm = a, so, r1 = r2, and the euclidian distance is 0
        assert_equal ( 0, o(c))

    def test_phase(self):
        o = specdesc("phase")
        c = cvec()
        assert_equal( 0., o(c))

    def test_kl(self):
        o = specdesc("kl")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        assert_almost_equal( sum(a * log(1.+ a/1.e-10 ) ) / o(c), 1., decimal=6)

    def test_mkl(self):
        o = specdesc("mkl")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        assert_almost_equal( sum(log(1.+ a/1.e-10 ) ) / o(c), 1, decimal=6)

    def test_specflux(self):
        o = specdesc("specflux")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        assert_equal( sum(a), o(c))
        assert_equal( 0, o(c))
        c.norm = zeros(c.length, dtype='float32')
        assert_equal( 0, o(c))

    def test_centroid(self):
        o = specdesc("centroid")
        c = cvec()
        # make sure centroid of zeros is zero
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        centroid = sum(a*a) / sum(a)
        assert_almost_equal (centroid, o(c), decimal = 2)

        c.norm = a * .5 
        assert_almost_equal (centroid, o(c), decimal = 2)

    def test_spread(self):
        o = specdesc("spread")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        centroid = sum(a*a) / sum(a)
        spread = sum( (a - centroid)**2 *a) / sum(a)
        assert_almost_equal (spread, o(c), decimal = 2)

        c.norm = a * 3
        assert_almost_equal (spread, o(c), decimal = 2)

    def test_skewness(self):
        o = specdesc("skewness")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        centroid = sum(a*a) / sum(a)
        spread = sum( (a - centroid)**2 *a) / sum(a)
        skewness = sum( (a - centroid)**3 *a) / sum(a) / spread **1.5
        assert_almost_equal (skewness, o(c), decimal = 2)

        c.norm = a * 3
        assert_almost_equal (skewness, o(c), decimal = 2)

    def test_kurtosis(self):
        o = specdesc("kurtosis")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length, dtype='float32')
        c.norm = a
        centroid = sum(a*a) / sum(a)
        spread = sum( (a - centroid)**2 *a) / sum(a)
        kurtosis = sum( (a - centroid)**4 *a) / sum(a) / spread **2
        assert_almost_equal (kurtosis, o(c), decimal = 2)

    def test_slope(self):
        o = specdesc("slope")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length * 2, 0, -2, dtype='float32')
        k = arange(c.length, dtype='float32')
        c.norm = a
        num = len(a) * sum(k*a) - sum(k)*sum(a)
        den = (len(a) * sum(k**2) - sum(k)**2)
        slope = num/den/sum(a)
        assert_almost_equal (slope, o(c), decimal = 5)

        a = arange(0, c.length * 2, +2, dtype='float32')
        c.norm = a
        num = len(a) * sum(k*a) - sum(k)*sum(a)
        den = (len(a) * sum(k**2) - sum(k)**2)
        slope = num/den/sum(a)
        assert_almost_equal (slope, o(c), decimal = 5)

        a = arange(0, c.length * 2, +2, dtype='float32')
        c.norm = a * 2
        assert_almost_equal (slope, o(c), decimal = 5)

    def test_decrease(self):
        o = specdesc("decrease")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length * 2, 0, -2, dtype='float32')
        k = arange(c.length, dtype='float32')
        c.norm = a
        decrease = sum((a[1:] - a [0]) / k[1:]) / sum(a[1:]) 
        assert_almost_equal (decrease, o(c), decimal = 5)

        a = arange(0, c.length * 2, +2, dtype='float32')
        c.norm = a
        decrease = sum((a[1:] - a [0]) / k[1:]) / sum(a[1:]) 
        assert_almost_equal (decrease, o(c), decimal = 5)

        a = arange(0, c.length * 2, +2, dtype='float32')
        c.norm = a * 2
        decrease = sum((a[1:] - a [0]) / k[1:]) / sum(a[1:]) 
        assert_almost_equal (decrease, o(c), decimal = 5)

    def test_rolloff(self):
        o = specdesc("rolloff")
        c = cvec()
        assert_equal( 0., o(c))
        a = arange(c.length * 2, 0, -2, dtype='float32')
        k = arange(c.length, dtype='float32')
        c.norm = a
        cumsum = .95*sum(a*a)
        i = 0; rollsum = 0
        while rollsum < cumsum:
          rollsum += a[i]*a[i]
          i+=1
        rolloff = i 
        assert_equal (rolloff, o(c))

if __name__ == '__main__':
    from unittest import main
    main()

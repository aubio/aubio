#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, cvec, pvoc
from numpy import array, shape
from numpy.random import random

precision = 6

class aubio_pvoc_test_case(TestCase):
    """ pvoc object test case """

    def test_members_automatic_sizes_default(self):
        """ check object creation with default parameters """
        f = pvoc()
        assert_equal ([f.win_s, f.hop_s], [1024, 512])

    def test_members_unnamed_params(self):
        """ check object creation with unnamed parameters """
        f = pvoc(2048, 128)
        assert_equal ([f.win_s, f.hop_s], [2048, 128])

    def test_members_named_params(self):
        """ check object creation with named parameters """
        f = pvoc(hop_s = 128, win_s = 2048)
        assert_equal ([f.win_s, f.hop_s], [2048, 128])

    def test_zeros(self):
        """ check the resynthesis of zeros gives zeros """
        win_s, hop_s = 1024, 256
        f = pvoc (win_s, hop_s)
        t = fvec (hop_s)
        for time in range( 4 * win_s / hop_s ):
            s = f(t)
            r = f.rdo(s)
            assert_equal ( array(t), 0)
            assert_equal ( s.norm, 0)
            assert_equal ( s.phas, 0)
            assert_equal ( r, 0)

    def test_resynth_two_steps(self):
        """ check the resynthesis of steps is correct with 50% overlap """
        hop_s = 512
        buf_s = hop_s * 2
        f = pvoc(buf_s, hop_s)
        sigin = fvec(hop_s)
        zeros = fvec(hop_s)
        # negative step
        sigin[20:50] = -.1
        # positive step
        sigin[100:200] = .1
        s1 = f(sigin)
        r1 = f.rdo(s1)
        s2 = f(zeros)
        r2 = f.rdo(s2)
        #self.plot_this ( s2.norm.T )
        assert_almost_equal ( r2, sigin, decimal = precision )
    
    def test_resynth_three_steps(self):
        """ check the resynthesis of steps is correct with 25% overlap """
        hop_s = 16
        buf_s = hop_s * 4
        sigin = fvec(hop_s)
        zeros = fvec(hop_s)
        f = pvoc(buf_s, hop_s)
        for i in xrange(hop_s):
            sigin[i] = random() * 2. - 1.
        t2 = f.rdo( f(sigin) )
        t2 = f.rdo( f(zeros) )
        t2 = f.rdo( f(zeros) )
        t2 = f.rdo( f(zeros) )
        assert_almost_equal( sigin, t2, decimal = precision )
    
    def plot_this( self, this ):
        from pylab import semilogy, show
        semilogy ( this )
        show ()

if __name__ == '__main__':
  from unittest import main
  main()


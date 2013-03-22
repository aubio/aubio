#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite
from numpy.testing import assert_equal, assert_almost_equal
from aubio import fvec, cvec, pvoc
from numpy import array, shape

class aubio_pvoc_test_case(TestCase):

  def test_members_automatic_sizes_default(self):
    f = pvoc()
    assert_equal ([f.win_s, f.hop_s], [1024, 512])

  def test_members_automatic_sizes_not_null(self):
    f = pvoc(2048, 128)
    assert_equal ([f.win_s, f.hop_s], [2048, 128])

  def test_zeros(self):
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

  def test_steps_two_channels(self):
    """ check the resynthesis of steps is correct """
    f = pvoc(1024, 512)
    t1 = fvec(512)
    t2 = fvec(512)
    # positive step in first channel
    t1[100:200] = .1
    # positive step in second channel
    t1[20:50] = -.1
    s1 = f(t1)
    r1 = f.rdo(s1)
    s2 = f(t2)
    r2 = f.rdo(s2)
    #self.plot_this ( s1.norm.T )
    assert_almost_equal ( t1, r2, decimal = 6 )
    
  def test_steps_three_random_channels(self):
    from random import random
    f = pvoc(64, 16)
    t0 = fvec(16)
    t1 = fvec(16)
    for i in xrange(16):
        t1[i] = random() * 2. - 1.
    t2 = f.rdo(f(t1))
    t2 = f.rdo(f(t0))
    t2 = f.rdo(f(t0))
    t2 = f.rdo(f(t0))
    assert_almost_equal( t1, t2, decimal = 6 )
    
  def plot_this( self, this ):
    from pylab import semilogy, show
    semilogy ( this )
    show ()

if __name__ == '__main__':
  from unittest import main
  main()


#! /usr/bin/env python

from numpy.testing import TestCase, run_module_suite

class aubiomodule_test_case(TestCase):

  def test_import(self):
    """ try importing aubio """
    import aubio 

if __name__ == '__main__':
  from unittest import main
  main()


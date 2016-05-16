#! /usr/bin/env python

from unittest import main
from numpy.testing import TestCase

class aubiomodule_test_case(TestCase):

    def test_import(self):
        """ try importing aubio """
        import aubio

if __name__ == '__main__':
    main()


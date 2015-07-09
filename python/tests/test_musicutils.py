#! /usr/bin/env python

from numpy.testing import TestCase
from aubio import window

class aubio_window(TestCase):

    def test_accept_name_and_size(self):
        window("default", 1024)

if __name__ == '__main__':
    from unittest import main
    main()

#! /usr/bin/env python

from numpy.testing import TestCase
from aubio import window

class aubio_window(TestCase):

    def test_accept_name_and_size(self):
        window("default", 1024)

    def test_fail_name_not_string(self):
        try:
            window(10, 1024)
        except ValueError, e:
            pass
        else:
            self.fail('non-string window type does not raise a ValueError')

    def test_fail_size_not_int(self):
        try:
            window("default", "default")
        except ValueError, e:
            pass
        else:
            self.fail('non-integer window length does not raise a ValueError')

if __name__ == '__main__':
    from unittest import main
    main()

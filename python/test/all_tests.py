#! /usr/bin/python

# add ${src}/python and ${src}/python/aubio/.libs to python path
# so the script is runnable from a compiled source tree.
import sys, os
sys.path.append('..')
sys.path.append(os.path.join('..','aubio','.libs'))

import unittest

modules_to_test = ('aubiomodule')

if __name__ == '__main__':
  for module in modules_to_test: exec('from %s import *' % module)
  unittest.main()

#! /usr/bin/python

# add ${src}/python and ${src}/python/aubio/.libs to python path
# so the script is runnable from a compiled source tree.
import sys, os
sys.path.append('..')
sys.path.append(os.path.join('..','aubio','.libs'))

import unittest

from glob import glob
modules_to_test = [i.split('.')[0] for i in glob('*.py')]

if __name__ == '__main__':
  for module in modules_to_test: 
    if module != 'all_tests': # (not actually needed)
      exec('from %s import *' % module)
  unittest.main()

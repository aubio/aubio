import unittest
from numpy import array

class aubio_unit_template(unittest.TestCase):
  """ a class derivated from unittest.TestCase """
  
  def assertCloseEnough(self, first, second, places=5, msg=None):
    """Fail if the two objects are unequal as determined by their
       *relative* difference rounded to the given number of decimal places
       (default 7) and comparing to zero.
    """
    if round(first, places) == 0:
      if round(second-first, places) != 0:
        raise self.failureException, \
              (msg or '%r != %r within %r places' % (first, second, places))
    else:
      if round((second-first)/first, places) != 0:
        raise self.failureException, \
              (msg or '%r != %r within %r places' % (first, second, places))

def array_from_text_file(filename, dtype = 'float'):
  return array([line.split() for line in open(filename).readlines()], 
      dtype = dtype)

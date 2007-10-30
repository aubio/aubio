import unittest

# this file is just to illustrates and test some of the unittest module
# functionalities.

class raise_test_case(unittest.TestCase):
  def test_assertEqual(self):
    """ check assertEqual returns AssertionError """
    try:
      self.assertEqual(0.,1.)
    except AssertionError:
      pass
    else:
      fail('expected an AssertionError exception')

  def test_assertAlmostEqual(self):
    """ check assertAlmostEqual returns AssertionError """
    try:
      self.assertAlmostEqual(0.,1.)
    except AssertionError:
      pass
    else:
      fail('expected an AssertionError exception')

  def test_assertRaises(self):
    """ check assertRaises works as expected """
    self.assertRaises(AssertionError, self.assertEqual, 0.,1.)
    self.assertRaises(AssertionError, self.assertAlmostEqual, 0.,1.,1)

if __name__ == '__main__':
  unittest.main()

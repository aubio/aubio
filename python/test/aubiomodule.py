import unittest

class aubiomodule_test_case(unittest.TestCase):

  def test_aubio(self):
    """ try importing aubio module """
    import aubio 

  def test_aubiowrapper(self):
    """ try importing aubio.aubiowrapper module """
    from aubio import aubiowrapper 
 
if __name__ == '__main__':
  unittest.main()

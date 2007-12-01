from template import aubio_unit_template
from localaubio import *

buf_size = 2048
channels = 3

class fvec_unit(aubio_unit_template):

  def setUp(self):
    self.vector = new_fvec(buf_size, channels)

  def tearDown(self):
    del_fvec(self.vector)

  def test_fvec(self):
    """ create and delete fvec """
    pass

  def test_fvec_read_sample(self):
    """ check new fvec elements are set to 0. """
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(0., fvec_read_sample(self.vector,channel,index))

  def test_fvec_write_sample(self):
    """ check new fvec elements are set with fvec_write_sample """
    for index in range(buf_size):
      for channel in range(channels):
        fvec_write_sample(self.vector,1.,channel,index)
    for index in range(buf_size):
      for channel in range(channels):
        self.assertEqual(1., fvec_read_sample(self.vector,channel,index))

if __name__ == '__main__':
  unittest.main()

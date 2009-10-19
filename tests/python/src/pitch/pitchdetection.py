from template import aubio_unit_template
from localaubio import *

buf_size = 4096
hop_size = 512
channels = 1
samplerate = 44100.

class pitch_test_case(unittest.TestCase):

  def setUp(self, type = aubio_pitch_yinfft, mode = aubio_pitchm_freq):
    self.create(type=type)
  
  def create(self, type = aubio_pitch_yinfft,
      mode = aubio_pitchm_freq):
    self.type = type
    self.o = new_aubio_pitch(buf_size, hop_size,
        channels, int(samplerate), type, mode)

  def tearDown(self):
    del_aubio_pitch(self.o)

  def test_pitch(self):
    """ create and delete pitch """
    pass

  def test_pitch_run_zeroes(self):
    """ run pitch on an empty buffer """
    vec = new_fvec(buf_size, channels)
    out = new_fvec(1, channels)
    for i in range(100):
      aubio_pitch_do(self.o,vec, out)
      self.assertEqual(fvec_read_sample(out, 0, 0),0.)
    del vec

  def test_pitch_run_4_impulses(self):
    """ run pitch on a train of 4 impulses """
    vec = new_fvec(buf_size, channels)
    out = new_fvec(1, channels)
    fvec_write_sample(vec,-1.,0,  0)
    fvec_write_sample(vec, 1.,0,  buf_size/4)
    fvec_write_sample(vec,-1.,0,  buf_size/2)
    fvec_write_sample(vec, 1.,0,3*buf_size/4)
    frequency = samplerate/2*4/buf_size
    for i in range(100):
      aubio_pitch_do(self.o,vec, out)
      self.assertEqual(fvec_read_sample(out, 0, 0),frequency)
    del vec

  def test_pitch_run_4_positive_impulses(self):
    """ run pitch on a train of 4 positive impulses of arbitrary size """
    vec = new_fvec(buf_size, channels)
    out = new_fvec(1, channels)
    frequency = samplerate/2*8/buf_size
    for i in range(100):
      fvec_write_sample(vec, 2.-.01*i,0,  0)
      fvec_write_sample(vec, 2.-.01*i,0,  buf_size/4)
      fvec_write_sample(vec, 2.-.01*i,0,  buf_size/2)
      fvec_write_sample(vec, 2.-.01*i,0,3*buf_size/4)
      aubio_pitch_do(self.o,vec, out)
      self.assertAlmostEqual(fvec_read_sample(out, 0, 0),frequency,1)
    del vec

  def test_pitch_run_4_negative_impulses(self):
    """ run pitch on a train of 4 negative impulses of arbitrary size """
    vec = new_fvec(buf_size, channels)
    out = new_fvec(1, channels)
    frequency = samplerate/2*8/buf_size
    for i in range(1,100):
      fvec_write_sample(vec,-.01*i,0,  0)
      fvec_write_sample(vec,-.01*i,0,  buf_size/4)
      fvec_write_sample(vec,-.01*i,0,  buf_size/2)
      fvec_write_sample(vec,-.01*i,0,3*buf_size/4)
      aubio_pitch_do(self.o,vec, out)
      self.assertAlmostEqual(fvec_read_sample(out, 0, 0),frequency,1)
    del vec

  def test_pitch_run_8_impulses(self):
    """ run pitch on a train of 8 impulses """
    vec = new_fvec(buf_size, channels)
    out = new_fvec(1, channels)
    fvec_write_sample(vec, 1.,0,  0)
    fvec_write_sample(vec,-1.,0,  buf_size/8)
    fvec_write_sample(vec, 1.,0,  buf_size/4)
    fvec_write_sample(vec,-1.,0,3*buf_size/8)
    fvec_write_sample(vec, 1.,0,  buf_size/2)
    fvec_write_sample(vec,-1.,0,5*buf_size/8)
    fvec_write_sample(vec, 1.,0,3*buf_size/4)
    fvec_write_sample(vec,-1.,0,7*buf_size/8)
    for i in range(100):
      aubio_pitch_do(self.o,vec, out)
      self.assertAlmostEqual(fvec_read_sample(out, 0, 0),
        samplerate/2/buf_size*8, 1) 
    del vec

"""
class pitch_yin_test_case(pitchdetection_test_case):
  def setUp(self, type = aubio_pitch_yin):
    self.create(type=type)

class pitch_fcomb_test_case(pitchdetection_test_case):
  def setUp(self, type = aubio_pitch_fcomb):
    self.create(type=type)

class pitch_mcomb_test_case(pitchdetection_test_case):
  def setUp(self, type = aubio_pitch_mcomb):
    self.create(type=type)

class pitch_schmitt_test_case(pitchdetection_test_case):
  def setUp(self, type = aubio_pitch_schmitt):
    self.create(type=type)
"""

if __name__ == '__main__':
  unittest.main()

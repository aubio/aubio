from template import *

import os.path

class aubiopitch_test_case(program_test_case):

  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = "PYTHONPATH=../../python:../../python/aubio/.libs " + \
              os.path.join('..','..','python','aubiopitch')

  def test_aubiopitch(self):
    """ test aubiopitch with default parameters """
    self.getOutput()
    # FIXME: useless check
    self.assertEqual(len(self.output.split('\n')), 1)
    #self.assertEqual(float(self.output.strip()), 0.)

  def test_aubiopitch_verbose(self):
    """ test aubiopitch with -v parameter """
    self.command += " -v "
    self.getOutput()
    # FIXME: loose checking: make sure at least 8 lines are printed
    assert len(self.output) >= 8

  def test_aubiopitch_devnull(self):
    """ test aubiopitch on /dev/null """
    self.filename = "/dev/null"
    # exit status should not be 0
    self.getOutput(expected_status = 256)
    # and there should be an error message
    assert len(self.output) > 0
    # that looks like this 
    output_lines = self.output.split('\n')
    #assert output_lines[0] == "Unable to open input file /dev/null."
    #assert output_lines[1] == "Supported file format but file is malformed."
    #assert output_lines[2] == "Could not open input file /dev/null."

mode_names = ["yinfft", "yin", "fcomb", "mcomb", "schmitt"]
for name in mode_names:
  exec("class aubiopitch_test_case_" + name + "(aubiopitch_test_case):\n\
    options = \" -m " + name + " \"")

class aubiopitch_test_yinfft(program_test_case):

  filename = os.path.join('..','..','sounds','16568__acclivity__TwoCows.wav')
  url = "http://www.freesound.org/samplesViewSingle.php?id=16568"
  progname = "PYTHONPATH=../../python:../../python/aubio/.libs " + \
              os.path.join('..','..','python','aubiopitch')
  options  = " -m yinfft -t 0.75 "

  def test_aubiopitch(self):
    """ test aubiopitch with default parameters """
    if not os.path.isfile(self.filename):
      print "Warning: file 16568_acclivity_TwoCows.wav was not found in %s" % os.path.dirname(self.filename) 
      print "download it from %s to actually run test" % url
      return
    self.getOutput()
    expected_output = open(os.path.join('examples','aubiopitch','yinfft'+'.'+os.path.basename(self.filename)+'.txt')).read()
    lines = 0
    for line_out, line_exp in zip(self.output.split('\n'), expected_output.split('\n')):
      try:
        assert line_exp == line_out, line_exp + " vs. " + line_out + " at line " + str(lines)
      except:
        open(os.path.join('examples','aubiopitch','yinfft'+'.'+os.path.basename(self.filename)+'.txt.out'),'w').write(self.output)
        raise
      lines += 1

if __name__ == '__main__': unittest.main()

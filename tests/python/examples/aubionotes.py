from template import *

class aubionotes_test_case(program_test_case):

  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = os.path.join('..','..','examples','aubionotes')

  def test_aubionotes(self):
    """ test aubionotes with default parameters """
    self.getOutput()
    # FIXME: useless check
    self.assertEqual(len(self.output.split('\n')), 1)
    self.assertEqual(float(self.output.strip()), 0.017415)

  def test_aubionotes_verbose(self):
    """ test aubionotes with -v parameter """
    self.command += " -v "
    self.getOutput()
    # FIXME: loose checking: make sure at least 8 lines are printed
    assert len(self.output) >= 8

  def test_aubionotes_devnull(self):
    """ test aubionotes on /dev/null """
    self.filename = "/dev/null"
    # exit status should not be 0
    self.getOutput(expected_status = 256)
    # and there should be an error message
    assert len(self.output) > 0
    # that looks like this 
    #output_lines = self.output.split('\n')
    #for line in output_lines:
    #  print line

mode_names = ["yinfft", "yin", "fcomb", "mcomb", "schmitt"]
for name in mode_names:
  exec("class aubionotes_test_case_" + name + "(aubionotes_test_case):\n\
    options = \" -p " + name + " \"")

if __name__ == '__main__': unittest.main()

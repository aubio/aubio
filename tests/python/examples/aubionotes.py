from template import *

class aubionotes_test_case(program_test_case):

  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = os.path.join('..','..','examples','aubioonset')

  def test_aubionotes(self):
    """ test aubionotes with default parameters """
    self.getOutput()
    # FIXME: useless check
    assert len(self.output) >= 0

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
    self.getOutput(expected_status = -1)
    assert self.status != 0
    # and there should be an error message
    assert len(self.output) > 0
    # that looks like this 
    output_lines = self.output.split('\n')
    assert output_lines[0] == "Unable to open input file /dev/null."
    #assert output_lines[1] == "Supported file format but file is malformed."
    assert output_lines[2] == "Could not open input file /dev/null."

if __name__ == '__main__': unittest.main()

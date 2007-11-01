import unittest
from commands import getstatusoutput

class program_test_case(unittest.TestCase):

  filename = "/dev/null"
  progname = "UNDEFINED"
  command = ""
  options = ""

  def getOutput(self, expected_status = 0):
    self.command = self.progname + ' -i ' + self.filename + self.command
    self.command += self.options
    [self.status, self.output] = getstatusoutput(self.command)
    if expected_status != -1:
      assert self.status == expected_status, \
        "expected status was %s, got %s\nOutput was:\n%s" % \
        (expected_status, self.status, self.output)

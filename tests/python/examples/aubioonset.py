from template import *

class aubioonset_test_case(program_test_case):
  
  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = os.path.join('..','..','examples','aubioonset')

  def test_aubioonset(self):
    """ test aubioonset with default parameters """
    self.getOutput()
    assert len(self.output) != 0, self.output
    assert len(str(self.output)) != 0, "no output produced with command:\n" \
      + self.command

  def test_aubioonset_with_inf_silence(self):
    """ test aubioonset with -s 0  """
    self.command += " -s 0" 
    self.getOutput()
    assert len(self.output) == 0, self.output

  def test_aubioonset_with_no_silence(self):
    """ test aubioonset with -s -100 """ 
    self.command += " -s -100 " 
    self.getOutput()
    # only one onset in woodblock.aiff
    assert len(self.output.split('\n')) == 1
    assert len(str(self.output)) != 0, "no output produced with command:\n" \
      + self.command
    # onset should be at 0.00000
    assert float(self.output.strip()) == 0.

class aubioonset_test_case_energy(aubioonset_test_case):
  def setUp(self, options = " -O energy "):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_specdiff(aubioonset_test_case):
  def setUp(self, options = " -O specdiff "):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_hfc(aubioonset_test_case):
  def setUp(self, options = " -O hfc "):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_complex(aubioonset_test_case):
  def setUp(self, options = " -O complex "):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_phase(aubioonset_test_case):
  def setUp(self, options = " -O phase"):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_kl(aubioonset_test_case):
  def setUp(self, options = " -O kl "):
    aubioonset_test_case.setUp(self, options = options)

class aubioonset_test_case_mkl(aubioonset_test_case):
  def setUp(self, options = " -O mkl "):
    aubioonset_test_case.setUp(self, options = options)

if __name__ == '__main__':

  unittest.main()

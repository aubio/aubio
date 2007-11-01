from template import *

class aubioonset_test_case(program_test_case):
  
  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = os.path.join('..','..','examples','aubioonset')

  def test_aubioonset(self):
    """ test aubioonset with default parameters """
    self.getOutput()
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
    assert len(str(self.output)) != 0, "no output produced with command:\n" \
      + self.command
    assert len(self.output.split('\n')) == 1
    # onset should be at 0.00000
    assert float(self.output.strip()) == 0.

for name in ["energy", "specdiff", "hfc", "complex", "phase", "kl", "mkl"]:
  exec("class aubioonset_test_case_"+name+"(aubioonset_test_case):\n\
  options = \" -O "+name+" \"")

if __name__ == '__main__': unittest.main()

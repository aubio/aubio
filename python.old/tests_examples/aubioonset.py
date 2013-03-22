from template import program_test_case

class aubioonset_unit(program_test_case):
  
  import os.path
  filename = os.path.join('..','..','sounds','woodblock.aiff')
  progname = os.path.join('..','..','examples','aubioonset')

  def test_aubioonset_with_inf_silence(self):
    """ test aubioonset with -s 0  """
    self.command += " -s 0" 
    self.getOutput()
    assert len(self.output) == 0, self.output

class aubioonset_unit_finds_onset(aubioonset_unit):

  def test_aubioonset(self):
    """ test aubioonset with default parameters """
    self.getOutput()
    assert len(str(self.output)) != 0, "no output produced with command:\n" \
      + self.command

  def test_aubioonset_with_no_silence(self):
    """ test aubioonset with -s -100 """ 
    self.command += " -s -100 " 
    self.getOutput()
    # only one onset in woodblock.aiff
    self.assertNotEqual(0, len(str(self.output)), \
      "no output produced with command:\n" + self.command)
    self.assertEqual(1, len(self.output.split('\n')) )
    # onset should be at 0.00000
    self.assertEqual(0, float(self.output.strip()))

list_of_onset_modes = ["energy", "specdiff", "hfc", "complex", "phase", \
                      "kl", "mkl", "specflux"]

for name in list_of_onset_modes:
  exec("class aubioonset_"+name+"_unit(aubioonset_unit):\n\
  options = \" -O "+name+" \"")

if __name__ == '__main__': unittest.main()

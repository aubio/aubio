#include "aubio.h"
#include "utils_tests.h"

int main (void)
{
  aubio_init();

  uint_t win_s = 32; // window size
  lvec_t * sp = new_lvec (win_s); // input buffer
  lvec_set_sample (sp, 2./3., 0);
  PRINT_MSG(AUBIO_LSMP_FMT "\n", lvec_get_sample (sp, 0));
  lvec_print (sp);
  lvec_ones (sp);
  lvec_print (sp);
  lvec_set_all (sp, 3./5.);
  lvec_print (sp);
  del_lvec(sp);

  aubio_cleanup();
  
  return 0;
}


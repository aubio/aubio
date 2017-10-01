#include <stdlib.h>
#include "aubio.h"

// When creating an aubio object, the user should check whether the object is
// set NULL, indicating the creation failed and the object was not allocated.

int main (void)
{
  aubio_init();

  uint_t return_code = 0;
  fvec_t *f = new_fvec(-12);
  cvec_t *c = new_cvec(-12);
  lvec_t *l = new_lvec(-12);
  aubio_fft_t *fft = new_aubio_fft(-12);
  if (f != NULL) {
    return_code = 1;
  } else if (c != NULL) {
    return_code = 2;
  } else if (l != NULL) {
    return_code = 3;
  } else if (fft != NULL) {
    return_code = 3;
  }

  aubio_cleanup();
  
  return return_code;
}

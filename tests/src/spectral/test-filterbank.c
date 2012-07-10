#define AUBIO_UNSTABLE 1

#include <stdio.h>
#include <aubio.h>

int
main (void)
{
  /* allocate some memory */
  uint_t win_s = 1024;          /* window size */
  uint_t n_filters = 13;        /* number of filters */
  cvec_t *in = new_cvec (win_s);      /* input buffer */
  fvec_t *out = new_fvec (win_s);     /* input buffer */
  fmat_t *coeffs = NULL;

  /* allocate fft and other memory space */
  aubio_filterbank_t *o = new_aubio_filterbank (n_filters, win_s);

  coeffs = aubio_filterbank_get_coeffs (o);
  if (coeffs == NULL) {
    return -1;
  }

  /*
  if (fvec_max (coeffs) != 0.) {
    return -1;
  }

  if (fvec_min (coeffs) != 0.) {
    return -1;
  }
  */

  fmat_print (coeffs);

  aubio_filterbank_do (o, in, out);

  del_aubio_filterbank (o);
  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

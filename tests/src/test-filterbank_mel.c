#define AUBIO_UNSTABLE 1

#include <stdio.h>
#include <aubio.h>

int
main (void)
{
  /* allocate some memory */
  uint_t win_s = 512;           /* fft size */
  uint_t n_filters = 40;        /* number of filters */
  cvec_t *in = new_cvec (win_s);      /* input buffer */
  fvec_t *out = new_fvec (win_s);     /* input buffer */
  fmat_t *coeffs = NULL;
  smpl_t samplerate = 16000.;

  /* allocate fft and other memory space */
  aubio_filterbank_t *o = new_aubio_filterbank (n_filters, win_s);

  /* assign Mel-frequency coefficients */
  aubio_filterbank_set_mel_coeffs_slaney (o, samplerate);

  coeffs = aubio_filterbank_get_coeffs (o);
  if (coeffs == NULL) {
    return -1;
  }

  //fmat_print (coeffs);

  //fprintf(stderr, "%f\n", fvec_sum(coeffs));

  aubio_filterbank_do (o, in, out);

  del_aubio_filterbank (o);
  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

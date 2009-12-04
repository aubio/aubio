#include <aubio.h>

int
main (void)
{
  /* allocate some memory */
  uint_t win_s = 512;           /* fft size */
  uint_t n_filters = 40;        /* number of filters */
  uint_t n_coefs = 13;          /* number of coefficients */
  cvec_t *in = new_cvec (win_s);      /* input buffer */
  fvec_t *out = new_fvec (n_coefs);     /* input buffer */
  smpl_t samplerate = 16000.;

  /* allocate fft and other memory space */
  aubio_mfcc_t *o = new_aubio_mfcc (win_s, n_filters, n_coefs, samplerate);

  cvec_set (in, 1.);

  aubio_mfcc_do (o, in, out);
  fvec_print (out);
  aubio_mfcc_do (o, in, out);
  fvec_print (out);

  del_aubio_mfcc (o);
  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

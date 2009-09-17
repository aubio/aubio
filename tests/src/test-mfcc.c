#include <aubio.h>

int
main (void)
{
  /* allocate some memory */
  uint_t win_s = 512;           /* fft size */
  uint_t channels = 1;          /* number of channel */
  uint_t n_filters = 40;        /* number of filters */
  uint_t n_coefs = 13;          /* number of coefficients */
  cvec_t *in = new_cvec (win_s, channels);      /* input buffer */
  fvec_t *out = new_fvec (n_coefs, channels);     /* input buffer */
  smpl_t samplerate = 16000.;
  uint_t i = 0;

  /* allocate fft and other memory space */
  aubio_mfcc_t *o = new_aubio_mfcc (win_s, samplerate, n_filters, n_coefs);

  for (i = 0; i < in->length; i ++) {
    in->norm[0][i] = 1.;
  }

  aubio_mfcc_do (o, in, out);

  fvec_print (out);

  del_aubio_mfcc (o);
  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

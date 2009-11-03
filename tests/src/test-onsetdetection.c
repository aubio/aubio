
#define AUBIO_UNSTABLE 1

#include <aubio.h>

int
main ()
{
  uint_t win_s = 1024;          /* window size */
  uint_t channels = 1;          /* number of channel */
  cvec_t *in = new_cvec (win_s, channels);      /* input buffer */
  fvec_t *out = new_fvec (1, channels); /* input buffer */

  aubio_specdesc_t *o;
  
  o = new_aubio_specdesc ("energy", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("energy", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("hfc", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("complex", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("phase", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("kl", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("mkl", win_s, channels);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

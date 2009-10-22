
#define AUBIO_UNSTABLE 1

#include <aubio.h>

int
main ()
{
  uint_t win_s = 1024;          /* window size */
  uint_t channels = 1;          /* number of channel */
  cvec_t *in = new_cvec (win_s, channels);      /* input buffer */
  fvec_t *out = new_fvec (1, channels); /* input buffer */

  aubio_onsetdetection_t *o;
  
  o = new_aubio_onsetdetection ("energy", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("energy", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("hfc", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("complex", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("phase", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("kl", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  o = new_aubio_onsetdetection ("mkl", win_s, channels);
  aubio_onsetdetection_do (o, in, out);
  del_aubio_onsetdetection (o);

  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

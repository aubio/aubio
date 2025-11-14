#include <aubio.h>
#include <stdio.h>

int main (void)
{
  uint_t win_s = 1024; // window size
  cvec_t *in = new_cvec (win_s); // input buffer
  fvec_t *out = new_fvec (1); // output spectral descriptor

  aubio_specdesc_t *o;

  o = new_aubio_specdesc ("energy", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("energy", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("hfc", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("complex", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("phase", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("kl", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  o = new_aubio_specdesc ("mkl", win_s);
  aubio_specdesc_do (o, in, out);
  del_aubio_specdesc (o);

  // Test rolloff with edge case (all energy in last bin)
  o = new_aubio_specdesc ("rolloff", win_s);
  cvec_zeros(in);
  in->norm[in->length - 1] = 1.0; // Put all energy in last bin
  aubio_specdesc_do (o, in, out);
  // Result should be the last bin index (length-1), not length
  if (out->data[0] >= in->length) {
    fprintf(stderr, "rolloff out of bounds: %f >= %d\n", out->data[0], in->length);
    return 1;
  }
  del_aubio_specdesc (o);

  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

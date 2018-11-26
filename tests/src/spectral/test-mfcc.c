#include <aubio.h>

int main (void)
{
  uint_t win_s = 512; // fft size
  uint_t n_filters = 40; // number of filters
  uint_t n_coeffs = 13; // number of coefficients
  smpl_t samplerate = 16000.; // samplerate
  cvec_t *in = new_cvec (win_s); // input buffer
  fvec_t *out = new_fvec (n_coeffs); // output coefficients

  if (new_aubio_mfcc(    0, n_filters, n_coeffs, samplerate)) return 1;
  if (new_aubio_mfcc(win_s,         0, n_coeffs, samplerate)) return 1;
  if (new_aubio_mfcc(win_s, n_filters,        0, samplerate)) return 1;
  if (new_aubio_mfcc(win_s, n_filters, n_coeffs,          0)) return 1;

  // create mfcc object
  aubio_mfcc_t *o = new_aubio_mfcc (win_s, n_filters, n_coeffs, samplerate);

  cvec_norm_set_all (in, 1.);
  aubio_mfcc_do (o, in, out);
  fvec_print (out);

  cvec_norm_set_all (in, .5);
  aubio_mfcc_do (o, in, out);
  fvec_print (out);

  // clean up
  del_aubio_mfcc (o);
  del_cvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

#include <aubio.h>

int main ()
{
  uint_t win_s = 1024; // window size
  uint_t n_filters = 13; // number of filters

  cvec_t *in_spec = new_cvec (win_s); // input vector of samples
  fvec_t *out_filters = new_fvec (n_filters); // per-band outputs
  fmat_t *coeffs; // pointer to the coefficients

  // create filterbank object
  aubio_filterbank_t *o = new_aubio_filterbank (n_filters, win_s);

  coeffs = aubio_filterbank_get_coeffs (o);

  aubio_filterbank_do (o, in_spec, out_filters);

  // fmat_print (coeffs);
  // cvec_print(in_spec);
  // fvec_print(out_filters);

  del_aubio_filterbank (o);
  del_cvec (in_spec);
  del_fvec (out_filters);
  aubio_cleanup ();

  return 0;
}

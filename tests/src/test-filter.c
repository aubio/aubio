#include <aubio.h>

int
main (void)
{
  /* allocate some memory */
  uint_t win_s = 32;            /* window size */
  uint_t channels = 1;          /* number of channel */
  fvec_t *in = new_fvec (win_s, channels);      /* input buffer */
  fvec_t *out = new_fvec (win_s, channels);     /* input buffer */


  aubio_filter_t *o = new_aubio_filter_c_weighting (channels, 44100);
  in->data[0][12] = 0.5;
  fvec_print (in);
  aubio_filter_do (o, in);
  fvec_print (in);
  del_aubio_filter (o);

  o = new_aubio_filter_c_weighting (channels, 44100);
  in->data[0][12] = 0.5;
  fvec_print (in);
  aubio_filter_do_outplace (o, in, out);
  fvec_print (out);
  del_aubio_filter (o);

  o = new_aubio_filter_c_weighting (channels, 44100);
  in->data[0][12] = 0.5;
  fvec_print (in);
  aubio_filter_do_filtfilt (o, in, out);
  fvec_print (out);
  del_aubio_filter (o);

  del_fvec (in);
  del_fvec (out);
  aubio_cleanup ();

  return 0;
}

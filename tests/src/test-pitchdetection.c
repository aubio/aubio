#include <aubio.h>

int
main ()
{
  /* allocate some memory */
  uint_t win_s = 1024;          /* window size */
  uint_t hop_s = win_s / 4;     /* hop size */
  uint_t samplerate = 44100;    /* samplerate */
  uint_t channels = 1;          /* number of channel */
  fvec_t *in = new_fvec (hop_s, channels);      /* input buffer */
  fvec_t *out = new_fvec (1, channels); /* input buffer */
  aubio_pitch_t *o =
      new_aubio_pitch ("default", win_s, hop_s, channels, samplerate);
  uint_t i = 0;

  while (i < 100) {
    aubio_pitch_do (o, in, out);
    i++;
  };

  del_aubio_pitch (o);
  del_fvec (in);
  aubio_cleanup ();

  return 0;
}

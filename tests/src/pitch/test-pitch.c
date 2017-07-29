#include <aubio.h>

int main (void)
{
  aubio_init();
  
  // 1. allocate some memory
  uint_t n = 0; // frame counter
  uint_t win_s = 1024; // window size
  uint_t hop_s = win_s / 4; // hop size
  uint_t samplerate = 44100; // samplerate
  // create some vectors
  fvec_t *input = new_fvec (hop_s); // input buffer
  fvec_t *out = new_fvec (1); // output candidates
  // create pitch object
  aubio_pitch_t *o = new_aubio_pitch ("default", win_s, hop_s, samplerate);

  // 2. do something with it
  while (n < 100) {
    // get `hop_s` new samples into `input`
    // ...
    // exectute pitch
    aubio_pitch_do (o, input, out);
    // do something with output candidates
    // ...
    n++;
  };

  // 3. clean up memory
  del_aubio_pitch (o);
  del_fvec (out);
  del_fvec (input);
  aubio_cleanup ();

  return 0;
}

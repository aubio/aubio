#include <aubio.h>

int main ()
{
  // 1. allocate some memory
  uint_t n = 0; // frame counter
  uint_t win_s = 1024; // window size
  uint_t hop_s = win_s / 4; // hop size
  uint_t samplerate = 44100; // samplerate
  // create some vectors
  fvec_t * input = new_fvec (win_s/4); // input buffer
  fvec_t * out = new_fvec (2); // input buffer
  // create onset object
  aubio_onset_t * onset = new_aubio_onset("complex", win_s, hop_s, samplerate);

  // 2. do something with it
  while (n < 10) {
    // get `hop_s` new samples into `input`
    // ...
    // exectute onset detection
    aubio_onset_do (onset, input, out);
    // do something with output candidates
    // ...
    n++;
  };

  // 3. clean up memory
  del_aubio_onset(onset);
  del_fvec(input);
  del_fvec(out);
  aubio_cleanup();

  return 0;
}

#include <aubio.h>

int main ()
{
  uint_t i = 0;
  uint_t win_s = 1024; // window size
  fvec_t * in = new_fvec (win_s); // input vector
  fvec_t * out = new_fvec (2); // output beat position

  // create tempo object
  aubio_tempo_t * o = new_aubio_tempo("complex", win_s, win_s/4, 44100.);

  smpl_t bpm, confidence;

  while (i < 1000) {
    // put some fresh data in input vector
    // ...

    // execute tempo
    aubio_tempo_do(o,in,out);
    // do something with the beats
    // ...

    // get bpm and confidence
    bpm = aubio_tempo_get_bpm(o);
    confidence = aubio_tempo_get_confidence(o);

    i++;
  };

  del_aubio_tempo(o);
  del_fvec(in);
  del_fvec(out);
  aubio_cleanup();

  return 0;
}

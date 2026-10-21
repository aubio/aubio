#define AUBIO_UNSTABLE 1

#include <aubio.h>
#include <stdio.h>

// not in a public header; defined in src/tempo/beattracking.c
uint_t fvec_gettimesig (fvec_t * acf, uint_t acflen, uint_t gp);

// Every period that fits must stay inside acf (aubio#434): a long period
// used to read up to 6 * gp + 1. Run under AddressSanitizer
// (sanitizers.yml) to catch a regression.
static int test_gettimesig (void)
{
  uint_t len = 64, gp, ts;
  fvec_t * acf = new_fvec (len);
  fvec_set_all (acf, 1.);
  for (gp = 0; gp < len; gp++) {
    ts = fvec_gettimesig (acf, len, gp);
    if (ts != 3 && ts != 4) return 1;
  }
  // energy at 3 periods wins, at 4 periods loses
  fvec_zeros (acf);
  acf->data[30] = 1.;
  if (fvec_gettimesig (acf, len, 10) != 3) return 1;
  fvec_zeros (acf);
  acf->data[40] = 1.;
  if (fvec_gettimesig (acf, len, 10) != 4) return 1;
  del_fvec (acf);
  return 0;
}

int main (void)
{
  uint_t i = 0;
  if (test_gettimesig ()) {
    fprintf(stderr, "fvec_gettimesig failed\n");
    return 1;
  }
  uint_t win_s = 1024; // window size
  fvec_t * in = new_fvec (win_s); // input buffer
  fvec_t * out = new_fvec (win_s / 4); // output beat position

  // create beattracking object
  aubio_beattracking_t * tempo  = new_aubio_beattracking(win_s, 256, 44100);

  smpl_t bpm, confidence;

  while (i < 10) {
    // put some fresh data in feature vector
    // ...

    aubio_beattracking_do(tempo,in,out);
    // do something  with the beats
    // ...

    // get bpm and confidence
    bpm = aubio_beattracking_get_bpm(tempo);
    confidence = aubio_beattracking_get_confidence(tempo);
    fprintf(stderr, "found bpm %f with confidence %f\n", bpm, confidence);
    i++;
  };

  del_aubio_beattracking(tempo);
  del_fvec(in);
  del_fvec(out);
  aubio_cleanup();

  return 0;
}

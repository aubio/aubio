#define AUBIO_UNSTABLE 1

// this file uses the unstable aubio api, please use aubio_pitch instead
// see src/pitch/pitch.h and tests/src/pitch/test-pitch.c

#include <aubio.h>

int main (void)
{
  uint_t n = 10; // compute n times
  uint_t win_s = 1024; // window size
  // create some vectors
  fvec_t * in = new_fvec (win_s); // input buffer
  fvec_t * out = new_fvec (1); // output candidates
  // create pitch object
  aubio_pitchyinfft_t *p  = new_aubio_pitchyinfft(44100, win_s);
  aubio_pitchyinfft_set_tolerance (p, (smpl_t)0.2);

  while ( n-- ) {
    aubio_pitchyinfft_do (p, in,out);
  };

  fvec_print(out);

  del_fvec(in);
  del_fvec(out);
  del_aubio_pitchyinfft(p);

  // Above 50.2kHz the spectrum reaches past the 25.1kHz end of the weighting
  // table; building the weights used to read past both tables (aubio#435).
  // Run under AddressSanitizer (sanitizers.yml) to catch a regression.
  {
    const uint_t rates[] = { 48000, 88200, 96000, 192000 };
    const uint_t sizes[] = { 256, 2048 };
    uint_t r, s;
    for (r = 0; r < sizeof(rates) / sizeof(rates[0]); r++) {
      for (s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        p = new_aubio_pitchyinfft(rates[r], sizes[s]);
        if (!p) return 1;
        del_aubio_pitchyinfft(p);
      }
    }
  }
  aubio_cleanup();

  return 0;
}

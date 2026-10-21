#include <aubio.h>
#include <math.h>
#include "utils_tests.h"

// gain in dB of the filter at freq, measured on a steady sine
static smpl_t gain_db (aubio_filter_t * f, uint_t samplerate, smpl_t freq)
{
  uint_t i, n = samplerate;
  fvec_t * v = new_fvec (n);
  smpl_t in = 0., out = 0.;
  for (i = 0; i < n; i++) {
    v->data[i] = (smpl_t)sin (2. * M_PI * freq * i / samplerate);
    if (i >= n / 2) in += v->data[i] * v->data[i];
  }
  aubio_filter_do_reset (f);
  aubio_filter_do (f, v);
  for (i = n / 2; i < n; i++) out += v->data[i] * v->data[i];
  del_fvec (v);
  return (smpl_t)(10. * log10 (out / in));
}

int main (void)
{
  aubio_filter_t * f;

  // any samplerate works: the shipped ones, LedFx's 30000, odd ones
  uint_t rates[] = { 4200, 8000, 16000, 22050, 30000, 44100, 96000, 192000 };
  uint_t nrates = sizeof(rates) / sizeof(rates[0]);
  uint_t samplerate = 0, i;

  for (i = 0; i < nrates; i++) {
    samplerate = rates[i];
    f = new_aubio_filter_a_weighting (samplerate);
    if (!f) return 1;
    // the curve is normalised to 0 dB at 1 kHz; near Nyquist the bilinear
    // transform's frequency warping moves it (+0.16 dB for A at 8000 Hz,
    // as with the coefficients aubio used to ship)
    if (samplerate >= 8000 && fabs (gain_db (f, samplerate, 1000.)) > 0.2) {
      PRINT_ERR ("A-weighting at %dHz: %f dB at 1 kHz\n", samplerate,
          gain_db (f, samplerate, 1000.));
      return 1;
    }
    del_aubio_filter (f);

    f = new_aubio_filter (7);
    if (aubio_filter_set_a_weighting (f, samplerate) != 0) return 1;
    del_aubio_filter (f);
  }

  // no samplerate
  EXPECT_LOGGED(f = new_aubio_filter_a_weighting (0));
  if (f) return 1;

  // order too small
  f = new_aubio_filter (2);
  EXPECT_LOGGED(if (aubio_filter_set_a_weighting (f, samplerate) == 0) return 1);
  del_aubio_filter (f);

  // order too big
  f = new_aubio_filter (12);
  EXPECT_LOGGED(if (aubio_filter_set_a_weighting (f, samplerate) == 0) return 1);
  del_aubio_filter (f);

  return 0;
}

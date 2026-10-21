/* Test file for NULL pointer allocation fixes (Issue #433)
 *
 * This test validates that all constructors properly handle allocation failures
 * by testing with small, valid parameters to ensure NULL checks don't break
 * normal operation.
 */

#define AUBIO_UNSTABLE 1
#include "aubio.h"
#include <stdio.h>
#include <stdlib.h>
#include "utils_tests.h"

int test_count = 0;
int test_passed = 0;

#define TEST(name, expr) do { \
  test_count++; \
  fprintf(stdout, "Testing %s... ", name); \
  if (expr) { \
    fprintf(stdout, "OK\n"); \
    test_passed++; \
  } else { \
    fprintf(stderr, "FAILED\n"); \
    return 1; \
  } \
} while(0)

/* Test core vector allocations */
int test_core_vectors(void) {
  fvec_t *fv = new_fvec(16);
  TEST("new_fvec", fv != NULL);
  del_fvec(fv);

  lvec_t *lv = new_lvec(16);
  TEST("new_lvec", lv != NULL);
  del_lvec(lv);

  cvec_t *cv = new_cvec(32);
  TEST("new_cvec", cv != NULL);
  del_cvec(cv);

  fmat_t *fm = new_fmat(4, 16);
  TEST("new_fmat", fm != NULL);
  del_fmat(fm);

  return 0;
}

/* Test filter constructors */
int test_filters(void) {
  aubio_filter_t *f;

  f = new_aubio_filter(5);
  TEST("new_aubio_filter(5)", f != NULL);
  del_aubio_filter(f);

  f = new_aubio_filter_a_weighting(44100);
  TEST("new_aubio_filter_a_weighting", f != NULL);
  del_aubio_filter(f);

  f = new_aubio_filter_c_weighting(44100);
  TEST("new_aubio_filter_c_weighting", f != NULL);
  del_aubio_filter(f);

  f = new_aubio_filter_biquad(0.1, 0.2, 0.1, 0.5, 0.0);
  TEST("new_aubio_filter_biquad", f != NULL);
  del_aubio_filter(f);

  return 0;
}

/* Test spectral constructors */
int test_spectral(void) {
  uint_t win_s = 512;
  uint_t hop_s = 256;

  aubio_specdesc_t *sd = new_aubio_specdesc("energy", win_s);
  TEST("new_aubio_specdesc", sd != NULL);
  del_aubio_specdesc(sd);

  aubio_pvoc_t *pv = new_aubio_pvoc(win_s, hop_s);
  TEST("new_aubio_pvoc", pv != NULL);
  del_aubio_pvoc(pv);

  aubio_mfcc_t *mfcc = new_aubio_mfcc(win_s, 40, 13, 44100);
  TEST("new_aubio_mfcc", mfcc != NULL);
  del_aubio_mfcc(mfcc);

  aubio_filterbank_t *fb = new_aubio_filterbank(40, win_s);
  TEST("new_aubio_filterbank", fb != NULL);
  del_aubio_filterbank(fb);

  aubio_spectral_whitening_t *sw = new_aubio_spectral_whitening(win_s, hop_s, 44100);
  TEST("new_aubio_spectral_whitening", sw != NULL);
  del_aubio_spectral_whitening(sw);

  aubio_fft_t *fft = new_aubio_fft(win_s);
  TEST("new_aubio_fft", fft != NULL);
  del_aubio_fft(fft);

  aubio_dct_t *dct = new_aubio_dct(40);
  TEST("new_aubio_dct", dct != NULL);
  del_aubio_dct(dct);

  aubio_tss_t *tss = new_aubio_tss(win_s, hop_s);
  TEST("new_aubio_tss", tss != NULL);
  del_aubio_tss(tss);

  return 0;
}

/* Test pitch detection constructors */
int test_pitch(void) {
  uint_t buf_size = 512;
  uint_t hop_size = 256;
  uint_t samplerate = 44100;

  aubio_pitch_t *p = new_aubio_pitch("default", buf_size, hop_size, samplerate);
  TEST("new_aubio_pitch", p != NULL);
  del_aubio_pitch(p);

  aubio_pitchyin_t *py = new_aubio_pitchyin(buf_size);
  TEST("new_aubio_pitchyin", py != NULL);
  del_aubio_pitchyin(py);

  aubio_pitchyinfft_t *pyf = new_aubio_pitchyinfft(samplerate, buf_size);
  TEST("new_aubio_pitchyinfft", pyf != NULL);
  del_aubio_pitchyinfft(pyf);

  aubio_pitchyinfast_t *pyfast = new_aubio_pitchyinfast(buf_size);
  TEST("new_aubio_pitchyinfast", pyfast != NULL);
  del_aubio_pitchyinfast(pyfast);

  aubio_pitchmcomb_t *pm = new_aubio_pitchmcomb(buf_size, hop_size);
  TEST("new_aubio_pitchmcomb", pm != NULL);
  del_aubio_pitchmcomb(pm);

  aubio_pitchfcomb_t *pf = new_aubio_pitchfcomb(buf_size, hop_size);
  TEST("new_aubio_pitchfcomb", pf != NULL);
  del_aubio_pitchfcomb(pf);

  aubio_pitchschmitt_t *ps = new_aubio_pitchschmitt(buf_size);
  TEST("new_aubio_pitchschmitt", ps != NULL);
  del_aubio_pitchschmitt(ps);

  aubio_pitchspecacf_t *psa = new_aubio_pitchspecacf(buf_size);
  TEST("new_aubio_pitchspecacf", psa != NULL);
  del_aubio_pitchspecacf(psa);

  return 0;
}

/* Test onset/tempo constructors */
int test_onset_tempo(void) {
  uint_t buf_size = 512;
  uint_t hop_size = 256;
  uint_t samplerate = 44100;

  aubio_onset_t *o = new_aubio_onset("default", buf_size, hop_size, samplerate);
  TEST("new_aubio_onset", o != NULL);
  del_aubio_onset(o);

  aubio_tempo_t *t = new_aubio_tempo("default", buf_size, hop_size, samplerate);
  TEST("new_aubio_tempo", t != NULL);
  del_aubio_tempo(t);

  aubio_peakpicker_t *pp = new_aubio_peakpicker();
  TEST("new_aubio_peakpicker", pp != NULL);
  del_aubio_peakpicker(pp);

  aubio_beattracking_t *bt = new_aubio_beattracking(512, hop_size, samplerate);
  TEST("new_aubio_beattracking", bt != NULL);
  del_aubio_beattracking(bt);

  return 0;
}

/* Test notes constructor */
int test_notes(void) {
  uint_t buf_size = 512;
  uint_t hop_size = 256;
  uint_t samplerate = 44100;

  aubio_notes_t *n = new_aubio_notes("default", buf_size, hop_size, samplerate);
  TEST("new_aubio_notes", n != NULL);
  del_aubio_notes(n);

  return 0;
}

/* Test I/O constructors */
int test_io(void) {
  aubio_sink_wavwrite_t *sw = new_aubio_sink_wavwrite("/tmp/test.wav", 44100);
  TEST("new_aubio_sink_wavwrite", sw != NULL);
  del_aubio_sink_wavwrite(sw);

  aubio_source_wavread_t *sr;
  EXPECT_LOGGED(sr = new_aubio_source_wavread("/nonexistent/aubio-null-alloc.wav", 44100, 256));
  TEST("new_aubio_source_wavread on a missing file returns NULL", sr == NULL);
  if (sr) del_aubio_source_wavread(sr);

  return 0;
}

/* Test synthesis constructors */
int test_synthesis(void) {
  aubio_sampler_t *s = new_aubio_sampler(44100, 256);
  TEST("new_aubio_sampler", s != NULL);
  del_aubio_sampler(s);

  aubio_wavetable_t *w = new_aubio_wavetable(44100, 256);
  TEST("new_aubio_wavetable", w != NULL);
  del_aubio_wavetable(w);

  return 0;
}

/* Test utility constructors */
int test_utils(void) {
  aubio_hist_t *h = new_aubio_hist(0.0, 1.0, 10);
  TEST("new_aubio_hist", h != NULL);
  del_aubio_hist(h);

  aubio_scale_t *sc = new_aubio_scale(0.0, 1.0, 0.0, 1.0);
  TEST("new_aubio_scale", sc != NULL);
  del_aubio_scale(sc);

  aubio_parameter_t *param = new_aubio_parameter(0.5, 0.0, 1.0);
  TEST("new_aubio_parameter", param != NULL);
  del_aubio_parameter(param);

  return 0;
}

#ifdef HAVE_SAMPLERATE
/* Test resampler if available */
int test_resampler(void) {
  aubio_resampler_t *r = new_aubio_resampler(0.5, 0);
  TEST("new_aubio_resampler", r != NULL);
  del_aubio_resampler(r);
  return 0;
}
#endif

/* Test invalid parameters are rejected */
int test_invalid_params(void) {
  aubio_filter_t *f;

  /* Test that order=0 is rejected */
  f = new_aubio_filter(0);
  TEST("new_aubio_filter(0) returns NULL", f == NULL);
  if (f) del_aubio_filter(f);

  /* Test that order > 512 is rejected */
  EXPECT_LOGGED(f = new_aubio_filter(1024));
  TEST("new_aubio_filter(1024) returns NULL", f == NULL);
  if (f) del_aubio_filter(f);

  /* Test negative size */
  fvec_t *fv = new_fvec(0);
  TEST("new_fvec(0) returns NULL", fv == NULL);
  if (fv) del_fvec(fv);

  return 0;
}

int main(void) {
  fprintf(stdout, "=== Testing NULL Pointer Allocation Fixes (Issue #433) ===\n\n");

  if (test_core_vectors()) return 1;
  if (test_filters()) return 1;
  if (test_spectral()) return 1;
  if (test_pitch()) return 1;
  if (test_onset_tempo()) return 1;
  if (test_notes()) return 1;
  if (test_io()) return 1;
  if (test_synthesis()) return 1;
  if (test_utils()) return 1;
#ifdef HAVE_SAMPLERATE
  if (test_resampler()) return 1;
#endif
  if (test_invalid_params()) return 1;

  fprintf(stdout, "\n=== Summary ===\n");
  fprintf(stdout, "Passed: %d/%d tests\n", test_passed, test_count);

  if (test_passed == test_count) {
    fprintf(stdout, "All allocation tests passed!\n");
    return 0;
  } else {
    fprintf(stderr, "Some tests failed!\n");
    return 1;
  }
}

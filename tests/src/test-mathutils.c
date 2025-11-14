#define AUBIO_UNSTABLE 1
#include "aubio.h"
#include "utils_tests.h"

int test_next_power_of_two (void);
int test_miditofreq (void);
int test_freqtomidi (void);
int test_aubio_window (void);
int test_quadratic_peak_mag_boundary (void);

int test_next_power_of_two (void)
{
  uint_t a, b;
  a = 15; b = aubio_next_power_of_two(a); assert(b == 16);
  fprintf(stdout, "aubio_next_power_of_two(%d) = %d\n", a, b);

  a = 17; b = aubio_next_power_of_two(a); assert(b == 32);
  fprintf(stdout, "aubio_next_power_of_two(%d) = %d\n", a, b);

  a = 31; b = aubio_next_power_of_two(a); assert(b == 32);
  fprintf(stdout, "aubio_next_power_of_two(%d) = %d\n", a, b);

  a = 32; b = aubio_next_power_of_two(a); assert(b == 32);
  fprintf(stdout, "aubio_next_power_of_two(%d) = %d\n", a, b);

  a = 33; b = aubio_next_power_of_two(a); assert(b == 64);
  fprintf(stdout, "aubio_next_power_of_two(%d) = %d\n", a, b);

  return 0;
}

int test_miditofreq (void)
{
  smpl_t a, b;
  fprintf(stdout, "b = aubio_miditofreq(a): [");
  for ( a = -123.; a < 400.; a += 20. ) {
    b = aubio_miditofreq(a);
    fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  }
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = -69.5;
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = -169.5;
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = 140.;
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = 0;
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = 8.2e10;
  b = aubio_miditofreq(a);
  fprintf(stdout, "(%.2f,  %.2f), ", a, b);
  a = -5.e10;
  fprintf(stdout, "(%.2f,  %.2f)", a, b);
  fprintf(stdout, "]\n");
  return 0;
}

int test_freqtomidi (void)
{
  smpl_t midi, freq;
  fprintf(stdout, "b = aubio_freqtomidi(a): [");
  for ( freq = 0.; freq < 30000.; freq += 440. ) {
    midi = aubio_freqtomidi(freq);
    fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  }
  freq = 69.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = -69.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = -169.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = 140.;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = 0;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = 8.2e10;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f), ", freq, midi);
  freq = -5.;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "(%.2f,  %.2f)]\n", freq, midi);
  return 0;
}

int test_aubio_window (void)
{
  uint_t window_size = 16;
  fvec_t * window = new_aubio_window("default", window_size);
  del_fvec(window);

  window = new_fvec(window_size);
  fvec_set_window(window, "rectangle");
  fvec_print(window);
  del_fvec(window);

  window_size /= 2.;
  window = new_aubio_window("parzen", window_size);
  fvec_print(window);
  del_fvec(window);

  window = new_aubio_window("rectangle", 16);
  del_fvec (window);
  return 0;
}

int test_quadratic_peak_mag_boundary (void)
{
  // Test that fvec_quadratic_peak_mag handles boundary conditions safely
  fvec_t *x = new_fvec(10);
  smpl_t mag;
  uint_t i;
  
  // Fill with test data - simple linear ramp
  for (i = 0; i < x->length; i++) {
    x->data[i] = (smpl_t)i + 1.0;  // Values 1.0 to 10.0
  }
  
  // Test at first position (boundary)
  // When pos=0, index=(uint_t)(0-0.5)+1 = 0+1 = 1
  // This should access x[0], x[1], x[2] which is safe
  mag = fvec_quadratic_peak_mag(x, 0.0);
  fprintf(stdout, "quadratic_peak_mag at pos 0.0 = %f (expected ~1.0)\n", mag);
  // The result will be interpolated, not necessarily x->data[0]
  
  // Test at position 1
  mag = fvec_quadratic_peak_mag(x, 1.0);
  fprintf(stdout, "quadratic_peak_mag at pos 1.0 = %f (expected ~2.0)\n", mag);
  
  // Test at last position (boundary)
  // When pos=9, index=(uint_t)(9-0.5)+1 = 8+1 = 9
  // This would try to access x[8], x[9], x[10] - x[10] is OOB!
  mag = fvec_quadratic_peak_mag(x, x->length - 1);
  fprintf(stdout, "quadratic_peak_mag at pos %d = %f (should not crash)\n", x->length - 1, mag);
  // Should return x->data[9] safely without accessing x[10]
  assert(mag == x->data[x->length - 1]);
  
  // Test near last position (fractional, should trigger bounds check)
  mag = fvec_quadratic_peak_mag(x, x->length - 1.5);
  fprintf(stdout, "quadratic_peak_mag at pos %f = %f (should not crash)\n", x->length - 1.5, mag);
  // Should return safely without crash
  
  // Test at valid middle position
  mag = fvec_quadratic_peak_mag(x, 5.0);
  fprintf(stdout, "quadratic_peak_mag at pos 5.0 = %f (expected ~6.0)\n", mag);
  
  // Test with fractional position in middle
  mag = fvec_quadratic_peak_mag(x, 5.5);
  fprintf(stdout, "quadratic_peak_mag at pos 5.5 = %f (should interpolate)\n", mag);
  // Should interpolate and not crash
  
  // Test out of bounds (negative)
  mag = fvec_quadratic_peak_mag(x, -1.0);
  fprintf(stdout, "quadratic_peak_mag at pos -1.0 = %f (expected 0.0)\n", mag);
  assert(mag == 0.0);
  
  // Test out of bounds (beyond length)
  mag = fvec_quadratic_peak_mag(x, x->length + 1.0);
  fprintf(stdout, "quadratic_peak_mag at pos %d = %f (expected 0.0)\n", x->length + 1, mag);
  assert(mag == 0.0);
  
  del_fvec(x);
  fprintf(stdout, "test_quadratic_peak_mag_boundary passed\n");
  return 0;
}

int main (void)
{
  test_next_power_of_two();
  test_miditofreq();
  test_freqtomidi();
  test_aubio_window();
  test_quadratic_peak_mag_boundary();
  return 0;
}

#include <stdio.h>
#include <assert.h>
#define AUBIO_UNSTABLE 1
#include <aubio.h>

int test_next_power_of_two()
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

int test_miditofreq()
{
  smpl_t midi, freq;
  for ( midi = 0; midi < 128; midi += 3 ) {
    freq = aubio_miditofreq(midi);
    fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  }
  midi = 69.5;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = -69.5;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = -169.5;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = 140.;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = 0;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = 8.2e10;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  midi = -5.e10;
  freq = aubio_miditofreq(midi);
  fprintf(stdout, "aubio_miditofreq(%.2f) = %.2f\n", midi, freq);
  return 0;
}

int test_freqtomidi()
{
  smpl_t midi, freq;
  for ( freq = 0.; freq < 30000.; freq += 440. ) {
    midi = aubio_freqtomidi(freq);
    fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  }
  freq = 69.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = -69.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = -169.5;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = 140.;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = 0;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = 8.2e10;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  freq = -5.;
  midi = aubio_freqtomidi(freq);
  fprintf(stdout, "aubio_freqtomidi(%.2f) = %.2f\n", freq, midi);
  return 0;
}

int test_aubio_window()
{
  uint_t window_size = 16;
  fvec_t * window = new_aubio_window("default", window_size);
  del_fvec(window);

  window = new_fvec(window_size);
  fvec_set_window(window, "rectangle");
  fvec_print(window);

  window_size /= 2.;
  window = new_aubio_window("triangle", window_size);
  fvec_print(window);
  del_fvec(window);

  window = new_aubio_window("rectangle", 16);
  del_fvec (window);
  return 0;
}

int main ()
{
  test_next_power_of_two();
  test_miditofreq();
  test_freqtomidi();
  return 0;
}

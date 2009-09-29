#include <aubio.h>

int main(){
  
  aubio_filter_t * f;

  uint_t rates[] = { 8000, 16000, 22050, 44100, 96000, 192000};
  uint_t nrates = 6;
  uint_t samplerate, i = 0;
  uint_t channels = 2;

  for ( samplerate = rates[i]; i < nrates ; i++ ) {
    f = new_aubio_filter_a_weighting (samplerate, channels);
    del_aubio_filter (f);

    f = new_aubio_filter (samplerate, 7, channels*2);
    aubio_filter_set_a_weighting (f);
    del_aubio_filter (f);
  }

  // samplerate unknown
  f = new_aubio_filter_a_weighting (12089, channels);
  del_aubio_filter (f);

  // order to small
  f = new_aubio_filter (samplerate, 2, channels*2);
  aubio_filter_set_a_weighting (f);
  del_aubio_filter (f);

  // order to big
  f = new_aubio_filter (samplerate, 12, channels*2);
  aubio_filter_set_a_weighting (f);
  del_aubio_filter (f);

  return 0;
}


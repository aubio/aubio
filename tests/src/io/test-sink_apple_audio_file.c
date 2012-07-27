#include <stdio.h>
#include <aubio.h>
#include "config.h"

char_t *path = "/Users/piem/archives/sounds/loops/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav";
char_t *outpath = "/var/tmp/test.wav";

int main(){
  int err = 0;
#ifdef __APPLE__
  uint_t samplerate = 44100;
  uint_t hop_size = 512;
  uint_t read = hop_size;
  fvec_t *vec = new_fvec(hop_size);
  aubio_source_apple_audio_t * i = new_aubio_source_apple_audio(path, samplerate, hop_size);
  aubio_sink_apple_audio_t *   o = new_aubio_sink_apple_audio(outpath, samplerate);

  if (!i || !o) { err = -1; goto beach; }

  while ( read == hop_size ) {
    aubio_source_apple_audio_do(i, vec, &read);
    aubio_sink_apple_audio_do(o, vec, read);
  }

beach:
  del_aubio_source_apple_audio(i);
  del_aubio_sink_apple_audio(o);
  del_fvec(vec);
#else
  fprintf(stderr, "ERR: aubio was not compiled with aubio_source_apple_audio\n");
#endif /* __APPLE__ */
  return err;
}


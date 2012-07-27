#include <stdio.h>
#include <aubio.h>
#include "config.h"

char_t *path = "/home/piem/archives/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav";
char_t *outpath = "/var/tmp/test.wav";

int main(){
  int err = 0;
  uint_t samplerate = 44100;
  uint_t hop_size = 512;
  uint_t read = hop_size;
  fvec_t *vec = new_fvec(hop_size);
  aubio_source_t * i = new_aubio_source(path, samplerate, hop_size);
  aubio_sink_t *   o = new_aubio_sink(outpath, samplerate);

  if (!i || !o) { err = -1; goto beach; }

  while ( read == hop_size ) {
    aubio_source_do(i, vec, &read);
    aubio_sink_do(o, vec, read);
  }

beach:
  del_aubio_source(i);
  del_aubio_sink(o);
  del_fvec(vec);
  return err;
}


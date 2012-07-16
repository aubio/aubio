#include <stdio.h>
#include <aubio.h>
#include "config.h"

char_t *path = "/home/piem/archives/samples/loops/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav";

int main(){
  int err = 0;
#ifdef HAVE_SNDFILE
  uint_t samplerate = 192000;
  uint_t hop_size = 512;
  uint_t read = hop_size;
  fvec_t *vec = new_fvec(hop_size);
  aubio_source_sndfile_t * s = new_aubio_source_sndfile(path, samplerate, hop_size);

  if (!s) { err = 1; goto beach; }

  while ( read == hop_size ) {
    aubio_source_sndfile_do(s, vec, &read);
    if (read == 0) break;
    fprintf(stdout, "%d [%f, %f, ..., %f]\n", read, vec->data[0], vec->data[1], vec->data[read - 1]);
  }

beach:
  del_aubio_source_sndfile(s);
  del_fvec(vec);
#else
  fprintf(stderr, "ERR: aubio was not compiled with aubio_source_sndfile\n");
  err = 2;
#endif /* HAVE_SNDFILE */
  return err;
}


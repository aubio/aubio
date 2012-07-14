#include <stdio.h>
#include <aubio.h>
#include "config.h"

char_t *path = "/home/piem/archives/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav";
char_t *outpath = "/var/tmp/test.wav";

int main(){
  int err = 0;
#ifdef HAVE_SNDFILE
  uint_t samplerate = 44100;
  uint_t hop_size = 512;
  uint_t read = hop_size;
  uint_t written = 512;
  fvec_t *vec = new_fvec(hop_size);
  aubio_source_sndfile_t * i = new_aubio_source_sndfile(path, samplerate, hop_size);
  aubio_sink_sndfile_t *   o = new_aubio_sink_sndfile(outpath, samplerate, hop_size);

  if (!i || !o) { err = -1; goto beach; }

  while ( read == hop_size ) {
    aubio_source_sndfile_do(i, vec, &read);
    if (read == 0) break;
    written = read;
    aubio_sink_sndfile_do(o, vec, &written);
    if (read != written)
      fprintf(stderr, "ERR: read %d, but wrote %d\n", read, written);
  }

beach:
  del_aubio_source_sndfile(i);
  del_aubio_sink_sndfile(o);
  del_fvec(vec);
#else
  fprintf(stderr, "ERR: aubio was not compiled with aubio_source_sndfile\n");
#endif /* HAVE_SNDFILE */
  return err;
}


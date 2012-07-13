#include <stdio.h>
#include <aubio.h>

char_t *path = "/Users/piem/archives/sounds/loops/drum_Chocolate_Milk_-_Ation_Speaks_Louder_Than_Words.wav";
//char_t *path = "/Users/piem/Downloads/Keziah Jones - Where's Life.mp3";

int main(){
  uint_t samplerate = 32000;
  uint_t hop_size = 1024;
  uint_t read = hop_size;
  fvec_t *vec = new_fvec(hop_size);
  aubio_source_t* s = new_aubio_source(path, samplerate, hop_size);

  if (!s) return -1;

  while ( read == hop_size ) {
    aubio_source_do(s, vec, &read);
    fprintf(stdout, "%d [%f, %f, ..., %f]\n", read, vec->data[0], vec->data[1], vec->data[read - 1]);
  }

  del_aubio_source(s);

  return 0;
}


#include <aubio.h>
#include "utils_tests.h"

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 3) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s <input_path> <output_path> [samplerate]\n", argv[0]);
    return err;
  }

  uint_t samplerate = 44100;
  uint_t hop_size = 512;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  if ( argc == 4 ) samplerate = atoi(argv[3]);

  fvec_t *vec = new_fvec(hop_size);
  aubio_source_t *i = new_aubio_source(source_path, samplerate, hop_size);
  if (samplerate == 0 ) samplerate = aubio_source_get_samplerate(i);
  aubio_sink_t *o = new_aubio_sink(sink_path, samplerate);

  if (!i || !o) { err = 1; goto beach; }

  do {
    aubio_source_do(i, vec, &read);
    aubio_sink_do(o, vec, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("wrote %d frames at %dHz from %s written to %s\n",
      n_frames, samplerate, source_path, sink_path);

beach:
  del_aubio_source(i);
  del_aubio_sink(o);
  del_fvec(vec);
  return err;
}

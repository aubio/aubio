#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

// this file uses the unstable aubio api, please use aubio_sink instead
// see src/io/sink.h and tests/src/sink/test-sink.c

int main (int argc, char **argv)
{
  aubio_init();

  sint_t err = 0;

  if (argc < 3) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s <input_path> <output_path> [samplerate] [hop_size]\n", argv[0]);
    return err;
  }

#ifdef HAVE_WAVWRITE
  uint_t samplerate = 0;
  uint_t hop_size = 512;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];

  if ( argc >= 4 ) samplerate = atoi(argv[3]);
  if ( argc >= 5 ) hop_size = atoi(argv[4]);
  if ( argc >= 6 ) {
    err = 2;
    PRINT_ERR("too many arguments\n");
    return err;
  }

  fvec_t *vec = new_fvec(hop_size);
  if (!vec) { err = 1; goto beach_fvec; }

  aubio_source_t *i = new_aubio_source(source_path, samplerate, hop_size);
  if (!i) { err = 1; goto beach_source; }

  if (samplerate == 0 ) samplerate = aubio_source_get_samplerate(i);

  aubio_sink_wavwrite_t *o = new_aubio_sink_wavwrite(sink_path, samplerate);
  if (!o) { err = 1; goto beach_sink; }

  do {
    aubio_source_do(i, vec, &read);
    aubio_sink_wavwrite_do(o, vec, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("read %d frames at %dHz (%d blocks) from %s written to %s\n",
      n_frames, samplerate, n_frames / hop_size,
      source_path, sink_path);

  del_aubio_sink_wavwrite(o);
beach_sink:
  del_aubio_source(i);
beach_source:
  del_fvec(vec);
beach_fvec:
#else
  err = 3;
  PRINT_ERR("aubio was not compiled with aubio_sink_wavwrite\n");
#endif /* HAVE_WAVWRITE */

  aubio_cleanup();
  
  return err;
}

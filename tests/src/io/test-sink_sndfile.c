#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "config.h"
#include "utils_tests.h"

// this file uses the unstable aubio api, please use aubio_sink instead
// see src/io/sink.h and tests/src/sink/test-sink.c

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 3) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s <input_path> <output_path> [samplerate]\n", argv[0]);
    return err;
  }

#ifdef HAVE_SNDFILE
  uint_t samplerate = 44100;
  uint_t hop_size = 512;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  if ( argc == 4 ) samplerate = atoi(argv[3]);

  fvec_t *vec = new_fvec(hop_size);
  aubio_source_sndfile_t * i = new_aubio_source_sndfile(source_path, samplerate, hop_size);
  if (samplerate == 0 ) samplerate = aubio_source_sndfile_get_samplerate(i);
  aubio_sink_sndfile_t *   o = new_aubio_sink_sndfile(sink_path, samplerate);

  if (!i || !o) { err = 1; goto beach; }

  do {
    aubio_source_sndfile_do(i, vec, &read);
    aubio_sink_sndfile_do(o, vec, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("%d frames read from %s\n written to %s at %dHz\n",
      n_frames, source_path, sink_path, samplerate);

beach:
  del_aubio_source_sndfile(i);
  del_aubio_sink_sndfile(o);
  del_fvec(vec);
#else
  err = 3;
  PRINT_ERR("aubio was not compiled with aubio_source_sndfile\n");
#endif /* HAVE_SNDFILE */
  return err;
}

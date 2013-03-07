#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"
#include "config.h"

// this file uses the unstable aubio api, please use aubio_source instead
// see src/io/source.h and tests/src/source/test-source.c

int main (int argc, char **argv)
{
  uint_t err = 0;
  if (argc < 2) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s <source_path> [samplerate]\n", argv[0]);
    return err;
  }

#ifdef HAVE_SNDFILE
  uint_t samplerate = 32000;
  uint_t hop_size = 256;
  uint_t n_frames = 0, read = 0;
  if ( argc == 3 ) samplerate = atoi(argv[2]);

  char_t *source_path = argv[1];

  fvec_t *vec = new_fvec(hop_size);
  aubio_source_sndfile_t * s = new_aubio_source_sndfile(source_path, samplerate, hop_size);
  if (samplerate == 0 ) samplerate = aubio_source_sndfile_get_samplerate(s);

  if (!s) { err = 1; goto beach; }

  do {
    aubio_source_sndfile_do(s, vec, &read);
    // fvec_print (vec);
    n_frames += read;
  } while ( read == hop_size );

beach:
  del_aubio_source_sndfile (s);
  del_fvec (vec);
#else
  err = 3;
  PRINT_ERR("aubio was not compiled with aubio_source_sndfile\n");
#endif /* HAVE_SNDFILE */
  return err;
}

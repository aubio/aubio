#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

// this file uses the unstable aubio api, please use aubio_source instead
// see src/io/source.h and tests/src/source/test-source.c

int main (int argc, char **argv)
{
  aubio_init();

  uint_t err = 0;
  if (argc < 2) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("read a wave file as a mono vector\n");
    PRINT_MSG("usage: %s <source_path> [samplerate] [hop_size]\n", argv[0]);
    PRINT_MSG("examples:\n");
    PRINT_MSG(" - read file.wav at original samplerate\n");
    PRINT_MSG("       %s file.wav\n", argv[0]);
    PRINT_MSG(" - read file.wav at 32000Hz\n");
    PRINT_MSG("       %s file.aif 32000\n", argv[0]);
    PRINT_MSG(" - read file.wav at original samplerate with 4096 blocks\n");
    PRINT_MSG("       %s file.wav 0 4096 \n", argv[0]);
    return err;
  }

#ifdef HAVE_SNDFILE
  uint_t samplerate = 0;
  uint_t hop_size = 256;
  uint_t n_frames = 0, read = 0;
  if ( argc == 3 ) samplerate = atoi(argv[2]);
  if ( argc == 4 ) hop_size = atoi(argv[3]);

  char_t *source_path = argv[1];


  aubio_source_sndfile_t * s =
    new_aubio_source_sndfile(source_path, samplerate, hop_size);
  if (!s) { err = 1; goto beach; }
  fvec_t *vec = new_fvec(hop_size);

  uint_t n_frames_expected = aubio_source_sndfile_get_duration(s);

  samplerate = aubio_source_sndfile_get_samplerate(s);

  do {
    aubio_source_sndfile_do(s, vec, &read);
    fvec_print (vec);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("read %d frames (expected %d) at %dHz (%d blocks) from %s\n",
            n_frames, n_frames_expected, samplerate, n_frames / hop_size,
            source_path);

  del_fvec (vec);
  del_aubio_source_sndfile (s);
beach:
#else
  err = 3;
  PRINT_ERR("aubio was not compiled with aubio_source_sndfile\n");
#endif /* HAVE_SNDFILE */

  aubio_cleanup();
  
  return err;
}

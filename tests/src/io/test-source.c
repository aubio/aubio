#include <aubio.h>
#include "utils_tests.h"

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

  uint_t samplerate = 0;
  uint_t hop_size = 256;
  uint_t n_frames = 0, read = 0;
  if ( argc == 3 ) samplerate = atoi(argv[2]);
  if ( argc == 4 ) hop_size = atoi(argv[3]);

  char_t *source_path = argv[1];


  aubio_source_t* s =
    new_aubio_source(source_path, samplerate, hop_size);
  if (!s) { err = 1; goto beach; }
  fvec_t *vec = new_fvec(hop_size);

  uint_t n_frames_expected = aubio_source_get_duration(s);

  samplerate = aubio_source_get_samplerate(s);

  do {
    aubio_source_do(s, vec, &read);
    fvec_print (vec);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("read %d frames (expected %d) at %dHz (%d blocks) from %s\n",
            n_frames, n_frames_expected, samplerate, n_frames / hop_size,
            source_path);

  // close the file (optional)
  aubio_source_close(s);
  // test closing the file a second time
  aubio_source_close(s);

  del_fvec (vec);
  del_aubio_source (s);
beach:
  aubio_cleanup();
  return err;
}

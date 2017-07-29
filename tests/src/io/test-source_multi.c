#include <aubio.h>
#include "utils_tests.h"

int main (int argc, char **argv)
{
  aubio_init();

  sint_t err = 0;
  if (argc < 2) {
    err = -2;
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
    PRINT_MSG(" - read file.wav at original samplerate with 256 frames blocks, mono\n");
    PRINT_MSG("       %s file.wav 0 4096 1\n", argv[0]);
    return err;
  }

  uint_t samplerate = 0;
  uint_t hop_size = 256;
  uint_t n_frames = 0, read = 0;
  uint_t n_channels = 0;
  if ( argc >= 3 ) samplerate = atoi(argv[2]);
  if ( argc >= 4 ) hop_size   = atoi(argv[3]);
  if ( argc >= 5 ) n_channels = atoi(argv[4]);

  char_t *source_path = argv[1];

  aubio_source_t* s = new_aubio_source(source_path, samplerate, hop_size);
  if (!s) { err = -1; goto beach; }

  if ( samplerate == 0 ) samplerate = aubio_source_get_samplerate(s);

  if ( n_channels == 0 ) n_channels = aubio_source_get_channels(s);

  fmat_t *mat = new_fmat(n_channels, hop_size);

  do {
    aubio_source_do_multi (s, mat, &read);
    fmat_print (mat);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("read %d frames in %d channels at %dHz (%d blocks) from %s\n",
      n_frames, n_channels, samplerate, n_frames / hop_size, source_path);

  del_fmat (mat);
  del_aubio_source (s);
beach:

  aubio_cleanup();
  
  return err;
}

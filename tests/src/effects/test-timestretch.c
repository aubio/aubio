#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 4) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s <input_path> <output_path> <stretch> [transpose] [mode] [hop_size] [samplerate]\n", argv[0]);
    PRINT_MSG(" with <stretch> a time stretching ratio in the range [0.025, 10.]\n");
    PRINT_MSG("      [transpose] a number of semi tones in the range [-24, 24]\n");
    PRINT_MSG("  and [mode] in 'default', 'crispness:0', ..., 'crispness:6'\n");
    return err;
  }

  uint_t samplerate = 0;
  uint_t hop_size = 64;
  smpl_t transpose = 0.;
  smpl_t stretch = 1.;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  char_t *mode = "default";

  stretch = atof(argv[3]);

  if ( argc >= 5 ) transpose = atof(argv[4]);
  if ( argc >= 6 ) mode = argv[5];
  if ( argc >= 7 ) hop_size = atoi(argv[6]);
  if ( argc >= 8 ) samplerate = atoi(argv[7]);
  if ( argc >= 9 ) {
    err = 2;
    PRINT_ERR("too many arguments\n");
    return err;
  }

  fvec_t *out = new_fvec(hop_size);
  if (!out) { err = 1; goto beach_fvec; }

  aubio_timestretch_t *ps = new_aubio_timestretch(source_path, mode,
      stretch, hop_size, samplerate);
  if (!ps) { err = 1; goto beach_timestretch; }
  if (samplerate == 0 ) samplerate = aubio_timestretch_get_samplerate(ps);

  aubio_sink_t *o = new_aubio_sink(sink_path, samplerate);
  if (!o) { err = 1; goto beach_sink; }

  if (transpose != 0) aubio_timestretch_set_transpose(ps, transpose);

#if 0
  do {
    if (aubio_timestretch_get_opened(ps) == 0)
      PRINT_MSG("not opened!\n");
    aubio_timestretch_get_opened(ps);
    aubio_timestretch_set_stretch(ps, stretch);
    aubio_timestretch_set_transpose(ps, transpose);
    aubio_timestretch_do(ps, out, &read);
    if (samplerate == 0) {
      PRINT_MSG("setting samplerate now to %d\n", aubio_timestretch_get_samplerate(ps));
      samplerate = aubio_timestretch_get_samplerate(ps);
      aubio_sink_preset_samplerate(o, samplerate);
      aubio_sink_preset_channels(o, 1);
    }
    aubio_sink_do(o, out, read);
    n_frames += read;
  } while ( read == hop_size );
#else

  aubio_timestretch_queue(ps, source_path, samplerate);

  do {
    aubio_timestretch_get_opened(ps);
    aubio_timestretch_set_stretch(ps, stretch);
    aubio_timestretch_set_transpose(ps, transpose);
    aubio_timestretch_do(ps, out, &read);
    if (n_frames == 34999 * hop_size) {
      PRINT_MSG("instant queuing?\n");
      aubio_timestretch_queue(ps, source_path, samplerate);
    }
    if (n_frames == 64999 * hop_size) {
      PRINT_MSG("instant queuing 2\n");
      aubio_timestretch_queue(ps, "/dev/null", samplerate);
    }
    if (n_frames == 54999 * hop_size) {
      PRINT_MSG("instant queuing?\n");
      aubio_timestretch_queue(ps, source_path, samplerate);
    }
    if (n_frames == 74999 * hop_size) {
      PRINT_MSG("instant queuing?\n");
      aubio_timestretch_queue(ps, source_path, samplerate);
    }
    aubio_sink_do(o, out, read);
  //} while ( read == hop_size );
    n_frames += hop_size;
  } while ( n_frames < 100000 * hop_size);
#endif

  PRINT_MSG("wrote %d frames at %dHz (%d blocks) from %s written to %s\n",
      n_frames, samplerate, n_frames / hop_size,
      source_path, sink_path);

  del_aubio_sink(o);
beach_sink:
  del_aubio_timestretch(ps);
beach_timestretch:
  del_fvec(out);
beach_fvec:
  return err;
}

#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

int test_wrong_params(void);

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 3) {
    PRINT_ERR("not enough arguments, running tests\n");
    err = test_wrong_params();
    PRINT_MSG("usage: %s <input_path> <output_path> [transpose] ", argv[0]);
    PRINT_MSG("[mode] [hop_size] [samplerate]\n");
    PRINT_MSG(" with [transpose] a number of semi tones in the range "
        " [-24, 24], and [mode] in 'default', 'crispness:0',"
        " 'crispness:1', ... 'crispness:6'\n");
    return err;
  }

#ifdef HAVE_RUBBERBAND
  uint_t samplerate = 0;
  uint_t hop_size = 64;
  smpl_t transpose = 0.;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  char_t *mode = "default";

  transpose = 0.;

  if ( argc >= 4 ) transpose = atof(argv[3]);
  if ( argc >= 5 ) mode = argv[4];
  if ( argc >= 6 ) hop_size = atoi(argv[5]);
  if ( argc >= 7 ) samplerate = atoi(argv[6]);

  fvec_t *vec = new_fvec(hop_size);
  fvec_t *out = new_fvec(hop_size);
  if (!vec) { err = 1; goto beach_fvec; }

  aubio_source_t *i = new_aubio_source(source_path, samplerate, hop_size);
  if (!i) { err = 1; goto beach_source; }

  if (samplerate == 0 ) samplerate = aubio_source_get_samplerate(i);

  aubio_sink_t *o = new_aubio_sink(sink_path, samplerate);
  if (!o) { err = 1; goto beach_sink; }

  aubio_pitchshift_t *ps =
    new_aubio_pitchshift(mode, transpose, hop_size, samplerate);
  if (!ps) { err = 1; goto beach_pitchshift; }

  do {
    aubio_source_do(i, vec, &read);
    //aubio_pitchshift_set_transpose(ps, tranpose);
    aubio_pitchshift_do(ps, vec, out);
    aubio_sink_do(o, out, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("read %d frames at %dHz (%d blocks) from %s\n",
      n_frames, samplerate, n_frames / hop_size,
      source_path);
  PRINT_MSG("wrote to %s with hop_size: %d, samplerate: %d, latency %d\n",
      sink_path, hop_size, samplerate, aubio_pitchshift_get_latency(ps));

  del_aubio_pitchshift(ps);
beach_pitchshift:
  del_aubio_sink(o);
beach_sink:
  del_aubio_source(i);
beach_source:
  del_fvec(vec);
  del_fvec(out);
beach_fvec:
  aubio_cleanup();
#else
  err = 0;
  PRINT_ERR("aubio was not compiled with rubberband\n");
#endif
  return err;
}

int test_wrong_params(void)
{
  const char_t *mode = "default";
  smpl_t transpose = 0.;
  uint_t hop_size = 256;
  uint_t samplerate = 44100;

  if (new_aubio_pitchshift("??", transpose, hop_size, samplerate)) return 1;
  if (new_aubio_pitchshift(mode,       28., hop_size, samplerate)) return 1;
  if (new_aubio_pitchshift(mode, transpose,        0, samplerate)) return 1;
  if (new_aubio_pitchshift(mode, transpose, hop_size,          0)) return 1;

  aubio_pitchshift_t *p = new_aubio_pitchshift(mode, transpose,
      hop_size, samplerate);
#ifdef HAVE_RUBBERBAND
  if (!p) return 1;
  if (!aubio_pitchshift_set_pitchscale(p, 0.1)) return 1;
  if (!aubio_pitchshift_set_transpose(p, -30)) return 1;
  del_aubio_pitchshift(p);
#else
  if (p) return 1;
#endif

  return run_on_default_source_and_sink(main);
}

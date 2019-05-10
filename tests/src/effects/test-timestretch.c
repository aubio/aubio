#define AUBIO_UNSTABLE 1
#include <aubio.h>
#include "utils_tests.h"

int test_wrong_params(void);

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 3 || argc >= 9) {
    PRINT_ERR("wrong number of arguments, running tests\n");
    err = test_wrong_params();
    PRINT_MSG("usage: %s <input_path> <output_path> <stretch> [transpose] [mode] [hop_size] [samplerate]\n", argv[0]);
    PRINT_MSG(" with <stretch> a time stretching ratio in the range [0.025, 10.]\n");
    PRINT_MSG("      [transpose] a number of semi tones in the range [-24, 24]\n");
    PRINT_MSG("  and [mode] in 'default', 'crispness:0', ..., 'crispness:6'\n");
    return err;
  }

#ifdef HAVE_RUBBERBAND
  uint_t samplerate = 0; // using source samplerate
  uint_t hop_size = 64;
  smpl_t transpose = 0.;
  smpl_t stretch = 1.;
  uint_t n_frames = 0, read = 0;
  uint_t eof = 0, source_read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  char_t *mode = "default";

  if ( argc >= 4 ) stretch = atof(argv[3]);
  if ( argc >= 5 ) transpose = atof(argv[4]);
  if ( argc >= 6 ) mode = argv[5];
  if ( argc >= 7 ) hop_size = atoi(argv[6]);
  if ( argc >= 8 ) samplerate = atoi(argv[7]);

  uint_t source_hopsize = 2048;
  aubio_source_t *s = new_aubio_source(source_path, samplerate, source_hopsize);
  if (!s) { err = 1; goto beach_source; }
  if (samplerate == 0) samplerate = aubio_source_get_samplerate(s);

  fvec_t *in = new_fvec(source_hopsize);
  fvec_t *out = new_fvec(hop_size);
  if (!out || !in) { err = 1; goto beach_fvec; }

  aubio_timestretch_t *ps = new_aubio_timestretch(mode, stretch, hop_size,
      samplerate);
  if (!ps) { err = 1; goto beach_timestretch; }
  //if (samplerate == 0 ) samplerate = aubio_timestretch_get_samplerate(ps);

  aubio_sink_t *o = new_aubio_sink(sink_path, samplerate);
  if (!o) { err = 1; goto beach_sink; }

  if (transpose != 0) aubio_timestretch_set_transpose(ps, transpose);

  do {
    //aubio_timestretch_set_stretch(ps, stretch);
    //aubio_timestretch_set_transpose(ps, transpose);

    while (aubio_timestretch_get_available(ps) < (sint_t)hop_size && !eof) {
      aubio_source_do(s, in, &source_read);
      aubio_timestretch_push(ps, in, source_read);
      if (source_read < in->length) eof = 1;
    }
#if 0
    if (n_frames == hop_size * 200) {
      PRINT_MSG("sampler: setting stretch gave %d\n",
          aubio_timestretch_set_stretch(ps, 2.) );
      PRINT_MSG("sampler: getting stretch gave %f\n",
          aubio_timestretch_get_stretch(ps) );
      PRINT_MSG("sampler: setting transpose gave %d\n",
          aubio_timestretch_set_transpose(ps, 12.) );
      PRINT_MSG("sampler: getting transpose gave %f\n",
          aubio_timestretch_get_transpose(ps) );
    }
#endif
    aubio_timestretch_do(ps, out, &read);
    aubio_sink_do(o, out, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("wrote %d frames at %dHz (%d blocks) from %s written to %s\n",
      n_frames, samplerate, n_frames / hop_size,
      source_path, sink_path);

  del_aubio_sink(o);
beach_sink:
  del_aubio_timestretch(ps);
beach_timestretch:
  del_fvec(out);
  del_fvec(in);
beach_fvec:
  del_aubio_source(s);
beach_source:
#else
  err = 0;
  PRINT_ERR("aubio was not compiled with rubberband\n");
#endif
  return err;
}

int test_wrong_params(void)
{
  const char_t *mode = "default";
  smpl_t stretch = 1.;
  uint_t hop_size = 256;
  uint_t samplerate = 44100;

  if (new_aubio_timestretch("ProcessOffline:?:", stretch, hop_size, samplerate)) return 1;
  if (new_aubio_timestretch("", stretch, hop_size, samplerate)) return 1;
  if (new_aubio_timestretch(mode,     41., hop_size, samplerate)) return 1;
  if (new_aubio_timestretch(mode, stretch,        0, samplerate)) return 1;
  if (new_aubio_timestretch(mode, stretch, hop_size,          0)) return 1;

  aubio_timestretch_t *p = new_aubio_timestretch(mode, stretch, hop_size,
      samplerate);
#ifdef HAVE_RUBBERBAND
  if (!p) return 1;

  if (aubio_timestretch_get_latency(p) == 0) return 1;

  if (aubio_timestretch_get_samplerate(p) != samplerate) return 1;

  aubio_timestretch_reset(p);

  if (aubio_timestretch_get_transpose(p) != 0) return 1;
  if (aubio_timestretch_set_transpose(p, 2.)) return 1;
  if (fabs(aubio_timestretch_get_transpose(p) - 2.) >= 1e-6) return 1;
  if (!aubio_timestretch_set_transpose(p, 200.)) return 1;
  if (!aubio_timestretch_set_transpose(p, -200.)) return 1;
  if (aubio_timestretch_set_transpose(p, 0.)) return 1;

  if (aubio_timestretch_get_pitchscale(p) != 1) return 1;
  if (aubio_timestretch_set_pitchscale(p, 2.)) return 1;
  if (fabs(aubio_timestretch_get_pitchscale(p) - 2.) >= 1e-6) return 1;
  if (!aubio_timestretch_set_pitchscale(p, 0.)) return 1;
  if (!aubio_timestretch_set_pitchscale(p, 6.)) return 1;

  if (aubio_timestretch_get_stretch(p) != stretch) return 1;
  if (aubio_timestretch_set_stretch(p, 2.)) return 1;
  if (fabs(aubio_timestretch_get_stretch(p) - 2.) >= 1e-6) return 1;
  if (!aubio_timestretch_set_stretch(p, 0.)) return 1;
  if (!aubio_timestretch_set_stretch(p, 41.)) return 1;

  del_aubio_timestretch(p);
#else
  if (p) return 1;
#endif

  return run_on_default_source_and_sink(main);
}

#include <string.h> // strncpy
#include <limits.h> // PATH_MAX
#include <aubio.h>
#include "utils_tests.h"

int time_was_reached (smpl_t time_s, uint_t n_frames, uint_t hop_size, uint_t samplerate) {
    if ((n_frames / hop_size) == (uint_t)(time_s * samplerate) / hop_size) {
      PRINT_MSG("reached %.2f sec at %d samples\n", time_s, n_frames);
      return 1;
    } else {
      return 0;
    }
}

int main (int argc, char **argv)
{
  sint_t err = 0;

  if (argc < 3) {
    PRINT_ERR("not enough arguments, running tests\n");
    err = run_on_default_source_and_sink(main);
    PRINT_MSG("usage: %s <input_path> <output_path> <sample_path> [samplerate] [blocksize]\n", argv[0]);
    return err;
  }

  uint_t samplerate = 44100; // default is 44100
  uint_t hop_size = 64; //256;
  uint_t n_frames = 0, frames_played = 0;
  uint_t read = 0;
  aubio_sink_t *sink = NULL;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];
  char_t sample_path[PATH_MAX];
  if ( argc >= 4 ) {
    strncpy(sample_path, argv[3], PATH_MAX - 1);
  } else {
    // use input_path as sample
    strncpy(sample_path, source_path, PATH_MAX - 1);
  }
  sample_path[PATH_MAX - 1] = '\0';
  if ( argc >= 5 ) samplerate = atoi(argv[4]);
  if ( argc >= 6 ) samplerate = atoi(argv[5]);

  fvec_t *vec = new_fvec(hop_size);

  aubio_sampler_t * sampler = new_aubio_sampler (hop_size, samplerate);
  if (!vec) goto beach;
  if (!sampler) goto beach_sampler;
  // load source file
  aubio_sampler_load (sampler, sample_path);
  // load source file (asynchronously)
  //aubio_sampler_queue (sampler, sample_path);
  samplerate = aubio_sampler_get_samplerate (sampler);
  if (samplerate == 0) {
    PRINT_ERR("starting with samplerate = 0\n");
    //goto beach_sink;
  }

  if (sink_path) {
    sink = new_aubio_sink(sink_path, samplerate);
    if (!sink) goto beach_sink;
  }

  smpl_t sample_duration = 2.953;
  uint_t sample_repeat = 10;
  smpl_t t1 = 1.,
         t2 = t1 + sample_duration * sample_repeat - .1,
         t3 = t2 - sample_duration + .1,
         t4 = t3 + sample_duration + .1,
         t5 = t4 + sample_duration + .1,
         total_duration = t5 + sample_duration + .1;

  //aubio_sampler_set_transpose(sampler, 0.);
  //aubio_sampler_set_stretch(sampler, .8);

  do {
    if (time_was_reached(t1, n_frames, hop_size, samplerate)) {
      PRINT_MSG("`-test one shot play of loaded sample\n");
      aubio_sampler_set_looping( sampler, 1);
      aubio_sampler_play ( sampler );
    } else if (time_was_reached(t2, n_frames, hop_size, samplerate)) {
      PRINT_MSG("`-test queueing while playing after eof was reached\n");
      //aubio_sampler_queue (sampler, sample_path);
      //aubio_sampler_play (sampler);
      aubio_sampler_set_looping( sampler, 0);
#if 0
    } else if (time_was_reached(t3, n_frames, hop_size, samplerate)) {
      PRINT_MSG("`-test queueing twice cancels the first one\n");
      aubio_sampler_queue (sampler, sample_path);
      aubio_sampler_queue (sampler, sample_path);
      aubio_sampler_play (sampler);
    } else if (time_was_reached(t4, n_frames, hop_size, samplerate)) {
      PRINT_MSG("`-test queueing a corrupt file\n");
      aubio_sampler_queue (sampler, "/dev/null");
      aubio_sampler_play (sampler);
    } else if (time_was_reached(t5, n_frames, hop_size, samplerate)) {
      aubio_sampler_stop ( sampler );
      PRINT_MSG("`-test queueing a correct file after a corrupt one\n");
      uint_t i;
      for (i = 0; i < 4; i++)
        aubio_sampler_queue (sampler, "/dev/null");
      aubio_sampler_queue (sampler, "/dev/null1");
      aubio_sampler_queue (sampler, "/dev/null2");
      aubio_sampler_queue (sampler, sample_path);
      aubio_sampler_play (sampler);
#endif
    }
    /*
    if (n_frames / hop_size == 40) {
      aubio_sampler_queue (sampler, sample_path);
      aubio_sampler_queue (sampler, sample_path);
      aubio_sampler_seek ( sampler, 0);
    }
    if (n_frames / hop_size == 70) {
      aubio_sampler_seek ( sampler, 0);
    }
    */
    aubio_sampler_do (sampler, vec, &read);
    if (sink) aubio_sink_do(sink, vec, hop_size);
    n_frames += hop_size;
    frames_played += read;
  //} while ( read == hop_size );
    // last for 40 seconds
  } while ( n_frames <= total_duration * samplerate );
  PRINT_MSG("reached %.2f sec at %d samples, sampler played %d frames\n",
      total_duration, n_frames, frames_played);

  if (sink) del_aubio_sink(sink);
beach_sink:
  del_aubio_sampler(sampler);
beach_sampler:
  del_fvec(vec);
beach:
  aubio_cleanup();
  return 0;
}

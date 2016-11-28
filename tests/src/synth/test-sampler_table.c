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

  if (argc < 1) {
    err = 2;
    PRINT_ERR("not enough arguments\n");
    PRINT_MSG("usage: %s [samplerate] [blocksize] [output_path]\n", argv[0]);
    return err;
  }

  uint_t samplerate = 44100;
  uint_t hop_size = 64; //256;
  uint_t n_frames = 0, frames_played = 0;
  uint_t read = 0;
  char_t *sink_path = NULL;
  aubio_sink_t *sink = NULL;

  if ( argc > 1 ) samplerate = atoi(argv[1]);
  if ( argc > 2 ) hop_size = atoi(argv[2]);
  if ( argc > 3 ) sink_path = argv[3];

  fvec_t *vec = new_fvec(hop_size);

  aubio_sampler_t * sampler = new_aubio_sampler (hop_size, samplerate);
  if (!vec) goto beach;
  if (!sampler) goto beach_sampler;
  // load source file
  //aubio_sampler_load (sampler, sample_path);
  // load source file (asynchronously)
  //aubio_sampler_queue (sampler, sample_path);
  samplerate = aubio_sampler_get_samplerate (sampler);
  if (samplerate == 0) {
    PRINT_ERR("starting with samplerate = 0\n");
    //goto beach_sink;
  }

  //fvec_t *table = new_fvec(1234560);
  fvec_t *table = new_fvec(123456);

  aubio_sampler_set_table(sampler, table);

  if (sink_path) {
    sink = new_aubio_sink(sink_path, samplerate);
    if (!sink) goto beach_sink;
  }

  smpl_t sample_duration = table->length/(smpl_t) samplerate;
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
      aubio_sampler_set_loop( sampler, 1);
      aubio_sampler_play ( sampler );
    } else if (time_was_reached(t2, n_frames, hop_size, samplerate)) {
      PRINT_MSG("`-test queueing while playing after eof was reached\n");
      //aubio_sampler_queue (sampler, sample_path);
      //aubio_sampler_play (sampler);
      aubio_sampler_set_loop( sampler, 0);
    }
    aubio_sampler_do (sampler, vec, &read);
    if (sink) aubio_sink_do(sink, vec, hop_size);
    n_frames += hop_size;
    frames_played += read;
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

// this should be included *after* custom functions have been defined

#ifndef aubio_sink_custom
#define aubio_sink_custom "undefined"
#endif /* aubio_sink_custom */

#ifdef HAVE_AUBIO_SINK_CUSTOM
int test_wrong_params(void);

int base_main(int argc, char **argv)
{
  uint_t err = 0;
  if (argc < 3 || argc >= 6) {
    PRINT_ERR("wrong number of arguments, running tests\n");
    err = test_wrong_params();
    PRINT_MSG("usage: %s <input_path> <output_path> [samplerate] [hop_size]\n",
        argv[0]);
    return err;
  }

  uint_t samplerate = 0;
  uint_t hop_size = 512;
  uint_t n_frames = 0, read = 0;

  char_t *source_path = argv[1];
  char_t *sink_path = argv[2];

  if ( argc >= 4 ) samplerate = atoi(argv[3]);
  if ( argc >= 5 ) hop_size = atoi(argv[4]);

  fvec_t *vec = new_fvec(hop_size);

  aubio_source_t *i = new_aubio_source(source_path, samplerate, hop_size);
  if (samplerate == 0 ) samplerate = aubio_source_get_samplerate(i);

  aubio_sink_custom_t *o = new_aubio_sink_custom(sink_path, samplerate);

  if (!vec || !i || !o) { err = 1; goto failure; }

  do {
    aubio_source_do(i, vec, &read);
    aubio_sink_custom_do(o, vec, read);
    n_frames += read;
  } while ( read == hop_size );

  PRINT_MSG("%d frames at %dHz (%d blocks) read from %s, wrote to %s\n",
      n_frames, samplerate, n_frames / hop_size,
      source_path, sink_path);

  // close sink now (optional)
  aubio_sink_custom_close(o);

failure:
  if (o)
    del_aubio_sink_custom(o);
  if (i)
    del_aubio_source(i);
  if (vec)
    del_fvec(vec);

  return err;
}

int test_wrong_params(void)
{
  return run_on_default_source_and_sink(base_main);
}

#else /* HAVE_AUBIO_SINK_CUSTOM */

int base_main(int argc, char** argv)
{
  PRINT_ERR("aubio was not compiled with aubio_sink_"
          aubio_sink_custom ", failed running %s with %d args\n",
          argv[0], argc);
  return 0;
}

#endif /* HAVE_AUBIO_SINK_CUSTOM */

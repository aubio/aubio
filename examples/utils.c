/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

/**

  This file includes some tools common to all examples. Code specific to the
  algorithm performed by each program should go in the source file of that
  program instead.

*/

#include "utils.h"

#ifdef HAVE_LASH
#include <lash/lash.h>
#include <pthread.h>
lash_client_t *aubio_lash_client;
lash_args_t *lash_args;
void *lash_thread_main (void *data);
int lash_main (void);
void save_data (void);
void restore_data (lash_config_t * lash_config);
pthread_t lash_thread;
#endif /* HAVE_LASH */

/* settings */
const char *output_filename = NULL;
const char *input_filename = NULL;
const char *onset_filename =
    AUBIO_PREFIX "/share/sounds/" PACKAGE "/woodblock.aiff";
int frames = 0;
int verbose = 0;
int usejack = 0;
int frames_delay = 0;


char_t * pitch_unit = "default";
char_t * pitch_mode = "default";

/* energy,specdiff,hfc,complexdomain,phase */
char_t * onset_mode = "default";
smpl_t threshold = 0.0;         // leave unset, only set as asked 
smpl_t silence = -90.;
uint_t buffer_size = 512;       //1024;
uint_t overlap_size = 256;      //512;
uint_t samplerate = 44100;


#ifdef HAVE_SNDFILE
aubio_sndfile_t *file = NULL;
aubio_sndfile_t *fileout = NULL;
#else
void *file = NULL;
void *fileout = NULL;
#endif

fvec_t *ibuf;
fvec_t *obuf;
fvec_t *woodblock;

/* badly redeclare some things */
smpl_t threshold;
smpl_t averaging;
const char *prog_name;

void flush_process (aubio_process_func_t process_func,
    aubio_print_func_t print);

void
usage (FILE * stream, int exit_code)
{
  fprintf (stream, "usage: %s [ options ] \n", prog_name);
  fprintf (stream,
      "       -h      --help          Display this message.\n"
      "       -v      --verbose       Be verbose.\n"
      "       -j      --jack          Use Jack.\n"
      "       -o      --output        Output type.\n"
      "       -i      --input         Input type.\n"
      "       -O      --onset         Select onset detection algorithm.\n"
      "       -t      --threshold     Set onset detection threshold.\n"
      "       -s      --silence       Select silence threshold.\n"
      "       -p      --pitch         Select pitch detection algorithm.\n"
      "       -B      --bufsize       Set buffer size.\n"
      "       -H      --hopsize       Set hopsize.\n"
      "       -a      --averaging     Use averaging.\n");
  exit (exit_code);
}

int
parse_args (int argc, char **argv)
{
  const char *options = "hvjo:i:O:t:s:p:B:H:a";
  int next_option;
  struct option long_options[] = {
    {"help", 0, NULL, 'h'},
    {"verbose", 0, NULL, 'v'},
    {"jack", 0, NULL, 'j'},
    {"output", 1, NULL, 'o'},
    {"input", 1, NULL, 'i'},
    {"onset", 1, NULL, 'O'},
    {"threshold", 1, NULL, 't'},
    {"silence", 1, NULL, 's'},
    {"pitch", 1, NULL, 'p'},
    {"averaging", 0, NULL, 'a'},
    {"bufsize", 1, NULL, 'B'},
    {"hopsize", 1, NULL, 'H'},
    {NULL, 0, NULL, 0}
  };
#ifdef HAVE_LASH
  lash_args = lash_extract_args (&argc, &argv);
#endif /* HAVE_LASH */
  prog_name = argv[0];
  if (argc < 1) {
    usage (stderr, 1);
    return -1;
  }
  do {
    next_option = getopt_long (argc, argv, options, long_options, NULL);
    switch (next_option) {
      case 'o':
        output_filename = optarg;
        break;
      case 'i':
        input_filename = optarg;
        break;
      case 'h':                /* help */
        usage (stdout, 0);
        return -1;
      case 'v':                /* verbose */
        verbose = 1;
        break;
      case 'j':
        usejack = 1;
        break;
      case 'O':                /*onset type */
        onset_mode = optarg;
        break;
      case 's':                /* silence value for onset */
        silence = (smpl_t) atof (optarg);
        break;
      case 't':                /* threshold value for onset */
        threshold = (smpl_t) atof (optarg);
        break;
      case 'p':
        pitch_mode = optarg;
        break;
      case 'a':
        averaging = 1;
        break;
      case 'B':
        buffer_size = atoi (optarg);
        break;
      case 'H':
        overlap_size = atoi (optarg);
        break;
      case '?':                /* unknown options */
        usage (stderr, 1);
        break;
      case -1:                 /* done with options */
        break;
      default:                 /*something else unexpected */
        fprintf (stderr, "Error parsing option '%c'\n", next_option);
        abort ();
    }
  }
  while (next_option != -1);

  if (input_filename != NULL) {
    debug ("Input file : %s\n", input_filename);
  } else if (input_filename != NULL && output_filename != NULL) {
    debug ("Input file : %s\n", input_filename);
    debug ("Output file : %s\n", output_filename);
  } else {
#if HAVE_JACK
    debug ("Jack input output\n");
    usejack = 1;
#else
    debug
        ("Error: Could not switch to jack mode\n   aubio was compiled without jack support\n");
    exit (1);
#endif
  }

  return 0;
}

#ifdef HAVE_SNDFILE

void
examples_common_init (int argc, char **argv)
{

  uint_t found_wood = 0;

  aubio_sndfile_t *onsetfile = NULL;
  /* parse command line arguments */
  parse_args (argc, argv);

  if (!usejack) {
    debug ("Opening files ...\n");
    file = new_aubio_sndfile_ro (input_filename);
    if (file == NULL) {
      outmsg ("Could not open input file %s.\n", input_filename);
      exit (1);
    }
    if (verbose)
      aubio_sndfile_info (file);
    samplerate = aubio_sndfile_samplerate (file);
    if (output_filename != NULL)
      fileout = new_aubio_sndfile_wo (file, output_filename);
  }
#ifdef HAVE_LASH
  else {
    aubio_lash_client = lash_init (lash_args, argv[0],
        LASH_Config_Data_Set | LASH_Terminal, LASH_PROTOCOL (2, 0));
    if (!aubio_lash_client) {
      fprintf (stderr, "%s: could not initialise lash\n", __FUNCTION__);
    }
    /* tell the lash server our client id */
    if (lash_enabled (aubio_lash_client)) {
      lash_event_t *event =
          (lash_event_t *) lash_event_new_with_type (LASH_Client_Name);
      lash_event_set_string (event, "aubio");
      lash_send_event (aubio_lash_client, event);
      pthread_create (&lash_thread, NULL, lash_thread_main, NULL);
    }
  }
#endif /* HAVE_LASH */

  woodblock = new_fvec (overlap_size);
  if (output_filename || usejack) {
    /* dummy assignement to keep egcs happy */
    found_wood = (onsetfile = new_aubio_sndfile_ro (onset_filename)) ||
        (onsetfile = new_aubio_sndfile_ro ("sounds/woodblock.aiff")) ||
        (onsetfile = new_aubio_sndfile_ro ("../sounds/woodblock.aiff"));
    if (onsetfile == NULL) {
      outmsg ("Could not find woodblock.aiff\n");
      exit (1);
    }
  }
  if (onsetfile) {
    /* read the output sound once */
    aubio_sndfile_read_mono (onsetfile, overlap_size, woodblock);
  }

  ibuf = new_fvec (overlap_size);
  obuf = new_fvec (overlap_size);

}

#else /* HAVE_SNDFILE */

void
examples_common_init (int argc, char **argv)
{
  outmsg ("Error, compiled without sndfile, nothing to do for now!\n");
}


#endif /* HAVE_SNDFILE */


void
examples_common_del (void)
{
#if HAVE_SNDFILE
  del_fvec (ibuf);
  del_fvec (obuf);
  del_fvec (woodblock);
#endif
  aubio_cleanup ();
}

#if HAVE_JACK
aubio_jack_t *jack_setup;
#endif

#if HAVE_SNDFILE

void
examples_common_process (aubio_process_func_t process_func,
    aubio_print_func_t print)
{
  if (usejack) {

#if HAVE_JACK
    debug ("Jack init ...\n");
    jack_setup = new_aubio_jack (1, 1,
        0, 1, (aubio_process_func_t) process_func);
    debug ("Jack activation ...\n");
    aubio_jack_activate (jack_setup);
    debug ("Processing (Ctrl+C to quit) ...\n");
    pause ();
    aubio_jack_close (jack_setup);
#else
    usage (stderr, 1);
    outmsg ("Compiled without jack output, exiting.\n");
#endif

  } else {
    /* phasevoc */
    debug ("Processing ...\n");

    frames = 0;

    while ((signed) overlap_size ==
        aubio_sndfile_read_mono (file, overlap_size, ibuf)) {
      process_func (&ibuf->data, &obuf->data, overlap_size);
      print ();
      if (output_filename != NULL) {
        aubio_sndfile_write (fileout, overlap_size, &obuf);
      }
      frames++;
    }

    debug ("Processed %d frames of %d samples.\n", frames, buffer_size);

    flush_process (process_func, print);
    del_aubio_sndfile (file);

    if (output_filename != NULL)
      del_aubio_sndfile (fileout);

  }
}

#else /* HAVE_SNDFILE */

void
examples_common_process (aubio_process_func_t process_func,
    aubio_print_func_t print)
{
}

#endif /* HAVE_SNDFILE */

void
flush_process (aubio_process_func_t process_func, aubio_print_func_t print)
{
  uint_t i;
  fvec_zeros(obuf);
  for (i = 0; (signed) i < frames_delay; i++) {
    process_func (&ibuf->data, &obuf->data, overlap_size);
    print ();
  }
}

void
send_noteon (int pitch, int velo)
{
  smpl_t mpitch = floor (aubio_freqtomidi (pitch) + .5);
#if HAVE_JACK
  jack_midi_event_t ev;
  ev.size = 3;
  ev.buffer = malloc (3 * sizeof (jack_midi_data_t)); // FIXME
  ev.time = 0;
  if (usejack) {
    ev.buffer[2] = velo;
    ev.buffer[1] = mpitch;
    if (velo == 0) {
      ev.buffer[0] = 0x80;      /* note off */
    } else {
      ev.buffer[0] = 0x90;      /* note on */
    }
    aubio_jack_midi_event_write (jack_setup, (jack_midi_event_t *) & ev);
  } else
#endif
  if (!verbose) {
    if (velo == 0) {
      outmsg ("%f\n", frames * overlap_size / (float) samplerate);
    } else {
      outmsg ("%f\t%f\t", mpitch, frames * overlap_size / (float) samplerate);
    }
  }
}


#if HAVE_LASH

void *
lash_thread_main (void *data __attribute__ ((unused)))
{
  printf ("LASH thread running\n");

  while (!lash_main ())
    usleep (1000);

  printf ("LASH thread finished\n");
  return NULL;
}

int
lash_main (void)
{
  lash_event_t *lash_event;
  lash_config_t *lash_config;

  while ((lash_event = lash_get_event (aubio_lash_client))) {
    switch (lash_event_get_type (lash_event)) {
      case LASH_Quit:
        lash_event_destroy (lash_event);
        exit (1);
        return 1;
      case LASH_Restore_Data_Set:
        lash_send_event (aubio_lash_client, lash_event);
        break;
      case LASH_Save_Data_Set:
        save_data ();
        lash_send_event (aubio_lash_client, lash_event);
        break;
      case LASH_Server_Lost:
        return 1;
      default:
        printf ("%s: received unknown LASH event of type %d",
            __FUNCTION__, lash_event_get_type (lash_event));
        lash_event_destroy (lash_event);
        break;
    }
  }

  while ((lash_config = lash_get_config (aubio_lash_client))) {
    restore_data (lash_config);
    lash_config_destroy (lash_config);
  }

  return 0;
}

void
save_data ()
{
  lash_config_t *lash_config;

  lash_config = lash_config_new_with_key ("threshold");
  lash_config_set_value_double (lash_config, threshold);
  lash_send_config (aubio_lash_client, lash_config);

}

void
restore_data (lash_config_t * lash_config)
{
  const char *lash_key;

  lash_key = lash_config_get_key (lash_config);

  if (strcmp (lash_key, "threshold") == 0) {
    threshold = lash_config_get_value_double (lash_config);
    return;
  }

}

#endif /* HAVE_LASH */

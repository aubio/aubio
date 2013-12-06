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
#ifdef HAVE_JACK
#include "jackio.h"
#endif /* HAVE_JACK */

int verbose = 0;
// input / output
char_t *sink_uri = NULL;
char_t *source_uri = NULL;
// general stuff
uint_t samplerate = 0;
uint_t buffer_size = 512;
uint_t overlap_size = 256;
// onset stuff
char_t * onset_method = "default";
smpl_t onset_threshold = 0.0; // will be set if != 0.
// pitch stuff
char_t * pitch_unit = "default";
char_t * pitch_method = "default";
smpl_t pitch_tolerance = 0.0; // will be set if != 0.
// tempo stuff
char_t * tempo_method = "default";
// more general stuff
smpl_t silence = -90.;
uint_t mix_input = 0;

//
// internal memory stuff
aubio_source_t *this_source = NULL;
aubio_sink_t *this_sink = NULL;
fvec_t *ibuf;
fvec_t *obuf;


/* settings */
int frames = 0;
int usejack = 0;
int frames_delay = 0;

extern void usage (FILE * stream, int exit_code);
extern int parse_args (int argc, char **argv);

void
examples_common_init (int argc, char **argv)
{

  /* parse command line arguments */
  parse_args (argc, argv);

  if (!usejack) {
    debug ("Opening files ...\n");
    this_source = new_aubio_source ((char_t*)source_uri, 0, overlap_size);
    if (this_source == NULL) {
      outmsg ("Could not open input file %s.\n", source_uri);
      exit (1);
    }
    samplerate = aubio_source_get_samplerate(this_source);
    if (sink_uri != NULL) {
      this_sink = new_aubio_sink ((char_t*)sink_uri, samplerate);
      if (this_sink == NULL) {
        outmsg ("Could not open output file %s.\n", sink_uri);
        exit (1);
      }
    }
  }
  ibuf = new_fvec (overlap_size);
  obuf = new_fvec (overlap_size);

}

void
examples_common_del (void)
{
  del_fvec (ibuf);
  del_fvec (obuf);
  aubio_cleanup ();
}

#if HAVE_JACK
aubio_jack_t *jack_setup;
#endif

void
examples_common_process (aubio_process_func_t process_func,
    aubio_print_func_t print)
{

  uint_t read = 0;
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

    do {
      aubio_source_do (this_source, ibuf, &read);
      process_func (&ibuf->data, &obuf->data, overlap_size);
      print ();
      if (this_sink) {
        aubio_sink_do (this_sink, obuf, overlap_size);
      }
      frames++;
    } while (read == overlap_size);

    debug ("Processed %d frames of %d samples.\n", frames, buffer_size);

    del_aubio_source (this_source);
    del_aubio_sink   (this_sink);

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


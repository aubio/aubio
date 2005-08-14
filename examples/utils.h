/*
	 Copyright (C) 2003 Paul Brossier

	 This program is free software; you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation; either version 2 of the License, or
	 (at your option) any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program; if not, write to the Free Software
	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
	 
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <unistd.h>
#include <math.h>
#include <aubio.h>
#include <aubioext.h>

#ifdef HAVE_C99_VARARGS_MACROS
#define debug(...)              if (verbose) fprintf (stderr, __VA_ARGS__)
#define errmsg(...)             fprintf (stderr, __VA_ARGS__)
#define outmsg(...)             fprintf (stdout, __VA_ARGS__)
#else
#define debug(format, args...)  if (verbose) fprintf(stderr, format , ##args)
#define errmsg(format, args...) fprintf(stderr, format , ##args)
#define outmsg(format, args...) fprintf(stdout, format , ##args)
#endif


extern int frames;
extern int verbose;
extern int usejack;
extern int usedoubled;
extern unsigned int median;
extern const char * output_filename;
extern const char * input_filename;
/* defined in utils.c */
void usage (FILE * stream, int exit_code);
int parse_args (int argc, char **argv);
void examples_common_init(int argc, char **argv);
void examples_common_del(void);
typedef void (aubio_print_func_t)(void);
#ifndef JACK_SUPPORT
typedef int (*aubio_process_func_t)
        (smpl_t **input, smpl_t **output, int nframes);
#endif
void examples_common_process(aubio_process_func_t process_func, aubio_print_func_t print);


void send_noteon(int pitch, int velo);
/** append new note candidate to the note_buffer and return filtered value. we
 * need to copy the input array as vec_median destroy its input data.*/
void note_append(fvec_t * note_buffer, smpl_t curnote); 
uint_t get_note(fvec_t *note_buffer, fvec_t *note_buffer2);

extern const char * output_filename;
extern const char * input_filename;
extern const char * onset_filename;
extern int verbose;
extern int usejack;
extern int usedoubled;


/* energy,specdiff,hfc,complexdomain,phase */
extern aubio_onsetdetection_type type_onset;
extern aubio_onsetdetection_type type_onset2;
extern smpl_t threshold;
extern smpl_t threshold2;
extern uint_t buffer_size;
extern uint_t overlap_size;
extern uint_t channels;
extern uint_t samplerate;


extern aubio_sndfile_t * file;
extern aubio_sndfile_t * fileout;

extern aubio_pvoc_t * pv;
extern fvec_t * ibuf;
extern fvec_t * obuf;
extern cvec_t * fftgrain;
extern fvec_t * woodblock;
extern aubio_onsetdetection_t *o;
extern aubio_onsetdetection_t *o2;
extern fvec_t *onset;
extern fvec_t *onset2;
extern int isonset;
extern aubio_pickpeak_t * parms;


/* pitch objects */
extern smpl_t pitch;
extern aubio_pitchdetection_t * pitchdet;
extern aubio_pitchdetection_type mode;
extern uint_t median;

extern fvec_t * note_buffer;
extern fvec_t * note_buffer2;
extern smpl_t curlevel;
extern smpl_t maxonset;

/* midi objects */
extern aubio_midi_player_t * mplay; 
extern aubio_midi_driver_t * mdriver; 
extern aubio_midi_event_t  * event;

extern smpl_t curnote;
extern smpl_t newnote;
extern uint_t isready;

/* per example param */
extern uint_t usepitch;


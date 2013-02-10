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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <unistd.h>
#include <math.h>               /* for isfinite */
#include <string.h>             /* for strcmp */
#include <aubio.h>
#include "config.h"
#ifdef HAVE_JACK
#include "jackio.h"
#endif /* HAVE_JACK */

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
extern int frames_delay;
/* defined in utils.c */
void usage (FILE * stream, int exit_code);
int parse_args (int argc, char **argv);
void examples_common_init (int argc, char **argv);
void examples_common_del (void);
typedef void (aubio_print_func_t) (void);
#ifndef HAVE_JACK
typedef int (*aubio_process_func_t)
  (smpl_t ** input, smpl_t ** output, int nframes);
#endif
void examples_common_process (aubio_process_func_t process_func,
    aubio_print_func_t print);

extern char_t * pitch_unit;
extern char_t * pitch_mode;

void send_noteon (int pitch, int velo);

extern const char *sink_uri;
extern char_t * onset_mode;
extern smpl_t threshold;
extern smpl_t silence;
extern int verbose;
extern int usejack;
extern uint_t buffer_size;
extern uint_t overlap_size;
extern uint_t samplerate;

extern fvec_t *ibuf;
extern fvec_t *obuf;
extern fvec_t *woodblock;

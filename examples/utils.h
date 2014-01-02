/*
  Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_C99_VARARGS_MACROS
#ifdef HAVE_DEBUG
#define debug(...)                fprintf (stderr, __VA_ARGS__)
#else
#define debug(...)
#endif
#define verbmsg(...)              if (verbose) fprintf (stderr, __VA_ARGS__)
#define errmsg(...)               fprintf (stderr, __VA_ARGS__)
#define outmsg(...)               fprintf (stdout, __VA_ARGS__)
#else
#ifdef HAVE_DEBUG
#define debug(...)                fprintf (stderr, format , **args)
#else
#define debug(...)
#endif
#define verbmsg(format, args...)  if (verbose) fprintf(stderr, format , ##args)
#define errmsg(format, args...)   fprintf(stderr, format , ##args)
#define outmsg(format, args...)   fprintf(stdout, format , ##args)
#endif

typedef void (aubio_print_func_t) (void);
void send_noteon (int pitch, int velo);

/** common process function */
typedef int (*aubio_process_func_t) (fvec_t * input, fvec_t * output);

void process_block (fvec_t *ibuf, fvec_t *obuf);
void process_print (void);

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

#ifndef JACKIO_H
#define JACKIO_H

/** 
 * @file
 *
 * Jack driver for aubio
 * 
 */

#ifdef __cplusplus
extern "C" {
#endif

/** jack object */
typedef struct _aubio_jack_t aubio_jack_t;
/** jack process function */
typedef int (*aubio_process_func_t)(smpl_t **input, smpl_t **output, int
    nframes);

/** jack device creation function */
aubio_jack_t * new_aubio_jack (uint_t inchannels, uint_t outchannels,
    aubio_process_func_t callback);
/** activate jack client (run jackprocess function) */
uint_t aubio_jack_activate(aubio_jack_t *jack_setup);
/** close and delete jack client */
void aubio_jack_close(aubio_jack_t *jack_setup);

#ifdef __cplusplus
}
#endif

#endif /* JACKIO_H */

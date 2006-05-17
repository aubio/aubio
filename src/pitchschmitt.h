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

/** \file 

   Pitch detection using a Schmitt trigger 
 
   This pitch extraction method implements a Schmitt trigger to estimate the
   period of a signal.

   This file was derived from the tuneit project, written by Mario Lang to
   detect the fundamental frequency of a sound.
   
   see http://delysid.org/tuneit.html 

*/

#ifndef _PITCHSCHMITT_H
#define _PITCHSCHMITT_H

#ifdef __cplusplus
extern "C" {
#endif

/** pitch detection object */
typedef struct _aubio_pitchschmitt_t aubio_pitchschmitt_t;

/** execute pitch detection on an input buffer 
 
  \param p pitch detection object as returned by new_aubio_pitchschmitt 
  \param input input signal window (length as specified at creation time) 
 
*/
smpl_t aubio_pitchschmitt_detect (aubio_pitchschmitt_t *p, fvec_t * input);
/** creation of the pitch detection object
 
  \param size size of the input buffer to analyse 
  \param samplerate sampling rate of the signal 
 
*/
aubio_pitchschmitt_t * new_aubio_pitchschmitt (uint_t size, uint_t samplerate);
/** deletion of the pitch detection object
 
  \param p pitch detection object as returned by new_aubio_pitchschmitt 
 
*/
void del_aubio_pitchschmitt (aubio_pitchschmitt_t *p);


#ifdef __cplusplus
}
#endif

#endif /* _PITCHSCHMITT_H */


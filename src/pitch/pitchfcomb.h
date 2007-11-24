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

   Pitch detection using a fast harmonic comb filter

   This pitch extraction method implements a fast harmonic comb filter to
   determine the fundamental frequency of a harmonic sound.

   This file was derived from the tuneit project, written by Mario Lang to
   detect the fundamental frequency of a sound.
   
   see http://delysid.org/tuneit.html 

*/

#ifndef _PITCHFCOMB_H
#define _PITCHFCOMB_H

#ifdef __cplusplus
extern "C" {
#endif

/** pitch detection object */
typedef struct _aubio_pitchfcomb_t aubio_pitchfcomb_t;

/** execute pitch detection on an input buffer 
 
  \param p pitch detection object as returned by new_aubio_pitchfcomb
  \param input input signal window (length as specified at creation time) 
 
*/
smpl_t aubio_pitchfcomb_detect (aubio_pitchfcomb_t *p, fvec_t * input);
/** creation of the pitch detection object
 
  \param bufsize size of the input buffer to analyse 
  \param hopsize step size between two consecutive analysis instant 
  \param samplerate sampling rate of the signal 
 
*/
aubio_pitchfcomb_t * new_aubio_pitchfcomb (uint_t bufsize, uint_t hopsize, uint_t samplerate);
/** deletion of the pitch detection object
 
  \param p pitch detection object as returned by new_aubio_pitchfcomb
 
*/
void del_aubio_pitchfcomb (aubio_pitchfcomb_t *p);


#ifdef __cplusplus
}
#endif

#endif /* _PITCHFCOMB_H */



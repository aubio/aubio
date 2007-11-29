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
  
  Pitch detection using the YIN algorithm
 
  This algorithm was developped by A. de Cheveigne and H. Kawahara and
  published in:
  
  De Cheveigné, A., Kawahara, H. (2002) "YIN, a fundamental frequency
  estimator for speech and music", J. Acoust. Soc. Am. 111, 1917-1930.  
 
  see http://recherche.ircam.fr/equipes/pcm/pub/people/cheveign.html

*/

#ifndef PITCHYIN_H
#define PITCHYIN_H

#ifdef __cplusplus
extern "C" {
#endif

/** compute difference function
  
  \param input input signal 
  \param yinbuf output buffer to store difference function (half shorter than input)

*/
void aubio_pitchyin_diff(fvec_t * input, fvec_t * yinbuf);

/** in place computation of the YIN cumulative normalised function 
  
  \param yinbuf input signal (a square difference function), also used to store function 

*/
void aubio_pitchyin_getcum(fvec_t * yinbuf);

/** detect pitch in a YIN function
  
  \param yinbuf input buffer as computed by aubio_pitchyin_getcum

*/
uint_t aubio_pitchyin_getpitch(fvec_t *yinbuf);

/** fast implementation of the YIN algorithm 
  
  \param input input signal 
  \param yinbuf input buffer used to compute the YIN function
  \param tol tolerance parameter for minima selection [default 0.15]

*/
smpl_t aubio_pitchyin_getpitchfast(fvec_t * input, fvec_t *yinbuf, smpl_t tol);

#ifdef __cplusplus
}
#endif

#endif /*PITCHYIN_H*/ 

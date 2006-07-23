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
 
  Pitch detection using a spectral implementation of the YIN algorithm
  
  This algorithm was derived from the YIN algorithm (see pitchyin.c). In this
  implementation, a Fourier transform is used to compute a tapered square
  difference function, which allows spectral weighting. Because the difference
  function is tapered, the selection of the period is simplified.
 
  Paul Brossier, ``Automatic annotation of musical audio for interactive
  systems'', Chapter 3, Pitch Analysis, PhD thesis, Centre for Digital music,
  Queen Mary University of London, London, UK, 2006.

*/

#ifndef PITCHYINFFT_H
#define PITCHYINFFT_H

#ifdef __cplusplus
extern "C" {
#endif

/** pitch detection object */
typedef struct _aubio_pitchyinfft_t aubio_pitchyinfft_t;

/** execute pitch detection on an input buffer 
 
  \param p pitch detection object as returned by new_aubio_pitchyinfft
  \param input input signal window (length as specified at creation time) 
  \param tol tolerance parameter for minima selection [default 0.85] 
 
*/
smpl_t aubio_pitchyinfft_detect (aubio_pitchyinfft_t *p, fvec_t * input, smpl_t tol);
/** creation of the pitch detection object
 
  \param bufsize size of the input buffer to analyse 
 
*/
aubio_pitchyinfft_t * new_aubio_pitchyinfft (uint_t bufsize);
/** deletion of the pitch detection object
 
  \param p pitch detection object as returned by new_aubio_pitchyinfft()
 
*/
void del_aubio_pitchyinfft (aubio_pitchyinfft_t *p);

#ifdef __cplusplus
}
#endif

#endif /*PITCHYINFFT_H*/ 

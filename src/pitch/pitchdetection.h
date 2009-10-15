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

#ifndef PITCHAUTOTCORR_H
#define PITCHAUTOTCORR_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Generic method for pitch detection

  This file creates the objects required for the computation of the selected
  pitch detection algorithm and output the results, in midi note or Hz.

*/

/** pitch detection object */
typedef struct _aubio_pitchdetection_t aubio_pitchdetection_t;

/** execute pitch detection on an input signal frame

  \param o pitch detection object as returned by new_aubio_pitchdetection()
  \param in input signal of size [hopsize x channels]
  \param out output pitch candidates of size [1 x channes]

*/
void aubio_pitchdetection_do (aubio_pitchdetection_t * o, fvec_t * in,
    fvec_t * out);

/** change yin or yinfft tolerance threshold

  \param o pitch detection object as returned by new_aubio_pitchdetection()
  \param tol tolerance default is 0.15 for yin and 0.85 for yinfft

*/
uint_t aubio_pitchdetection_set_tolerance (aubio_pitchdetection_t * o,
    smpl_t tol);

/** deletion of the pitch detection object

  \param o pitch detection object as returned by new_aubio_pitchdetection()

*/
void del_aubio_pitchdetection (aubio_pitchdetection_t * o);

/** creation of the pitch detection object

  \param mode set pitch detection algorithm
  \param bufsize size of the input buffer to analyse
  \param hopsize step size between two consecutive analysis instant
  \param channels number of channels to analyse
  \param samplerate sampling rate of the signal

*/
aubio_pitchdetection_t *new_aubio_pitchdetection (char_t * mode,
    uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate);

/** set the output unit of the pitch detection object 

  \param o pitch detection object as returned by new_aubio_pitchdetection()
  \param mode set pitch units for output

*/
uint_t aubio_pitchdetection_set_unit (aubio_pitchdetection_t * o,
    char_t * mode);

#ifdef __cplusplus
}
#endif

#endif /*PITCHDETECTION_H*/

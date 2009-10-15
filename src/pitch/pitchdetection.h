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

  \param p pitch detection object as returned by new_aubio_pitchdetection
  \param ibuf input signal of length hopsize

*/
void aubio_pitchdetection_do (aubio_pitchdetection_t * p, fvec_t * ibuf, fvec_t * obuf);

/** change yin or yinfft tolerance threshold

  default is 0.15 for yin and 0.85 for yinfft

*/
void aubio_pitchdetection_set_tolerance(aubio_pitchdetection_t *p, smpl_t tol);

/** deletion of the pitch detection object

  \param p pitch detection object as returned by new_aubio_pitchdetection

*/
void del_aubio_pitchdetection(aubio_pitchdetection_t * p);

/** creation of the pitch detection object

  \param bufsize size of the input buffer to analyse
  \param hopsize step size between two consecutive analysis instant
  \param channels number of channels to analyse
  \param samplerate sampling rate of the signal
  \param type set pitch detection algorithm
  \param mode set pitch units for output

*/
aubio_pitchdetection_t *new_aubio_pitchdetection (char_t * pitch_mode,
    uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate);

/** set the output unit of the pitch detection object */
uint_t aubio_pitchdetection_set_unit (aubio_pitchdetection_t *p, char_t * pitch_unit);

#ifdef __cplusplus
}
#endif

#endif /*PITCHDETECTION_H*/

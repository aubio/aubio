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

#ifndef PITCH_H
#define PITCH_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Generic method for pitch detection

  This file creates the objects required for the computation of the selected
  pitch detection algorithm and output the results, in midi note or Hz.

  \example pitch/test-pitch.c

*/

/** pitch detection object */
typedef struct _aubio_pitch_t aubio_pitch_t;

/** execute pitch detection on an input signal frame

  \param o pitch detection object as returned by new_aubio_pitch()
  \param in input signal of size [hop_size]
  \param out output pitch candidates of size [1]

*/
void aubio_pitch_do (aubio_pitch_t * o, fvec_t * in, fvec_t * out);

/** change yin or yinfft tolerance threshold

  \param o pitch detection object as returned by new_aubio_pitch()
  \param tol tolerance default is 0.15 for yin and 0.85 for yinfft

*/
uint_t aubio_pitch_set_tolerance (aubio_pitch_t * o, smpl_t tol);

/** deletion of the pitch detection object

  \param o pitch detection object as returned by new_aubio_pitch()

*/
void del_aubio_pitch (aubio_pitch_t * o);

/** creation of the pitch detection object

  \param method set pitch detection algorithm
  \param buf_size size of the input buffer to analyse
  \param hop_size step size between two consecutive analysis instant
  \param samplerate sampling rate of the signal

*/
aubio_pitch_t *new_aubio_pitch (char_t * method,
    uint_t buf_size, uint_t hop_size, uint_t samplerate);

/** set the output unit of the pitch detection object 

  \param o pitch detection object as returned by new_aubio_pitch()
  \param mode set pitch units for output

*/
uint_t aubio_pitch_set_unit (aubio_pitch_t * o, char_t * mode);

/** get the current confidence

  \param o pitch detection object as returned by new_aubio_pitch()
  \return the current confidence of the pitch algorithm

The confidence

*/
smpl_t aubio_pitch_get_confidence (aubio_pitch_t * o);

#ifdef __cplusplus
}
#endif

#endif /*PITCH_H*/

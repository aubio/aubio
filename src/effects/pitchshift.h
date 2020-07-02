/*
  Copyright (C) 2016 Paul Brossier <piem@aubio.org>

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

#ifndef AUBIO_PITCHSHIFT_H
#define AUBIO_PITCHSHIFT_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Pitch shifting object

  ::aubio_pitchshift_t can be used to transpose a stream of blocks of frames.

  \example effects/test-pitchshift.c

*/

/** pitch shifting object */
typedef struct _aubio_pitchshift_t aubio_pitchshift_t;

/** execute pitch shifting on an input signal frame

  \param o pitch shifting object as returned by new_aubio_pitchshift()
  \param in input signal of size [hop_size]
  \param out output pitch candidates of size [1]

*/
void aubio_pitchshift_do (aubio_pitchshift_t * o, const fvec_t * in,
        fvec_t * out);

/** deletion of the pitch shifting object

  \param o pitch shifting object as returned by new_aubio_pitchshift()

*/
void del_aubio_pitchshift (aubio_pitchshift_t * o);

/** creation of the pitch shifting object

  \param method set pitch shifting algorithm ("default")
  \param transpose initial pitch transposition
  \param hop_size step size between two consecutive analysis instant
  \param samplerate sampling rate of the signal

  \return newly created ::aubio_pitchshift_t

*/
aubio_pitchshift_t *new_aubio_pitchshift (const char_t * method,
    smpl_t transpose, uint_t hop_size, uint_t samplerate);

/** get the latency of the pitch shifting object, in samples

  \param o pitch shifting object as returned by ::new_aubio_pitchshift()

  \return latency of the pitch shifting object in samples

*/
uint_t aubio_pitchshift_get_latency (aubio_pitchshift_t * o);

/** set the pitch scale of the pitch shifting object

  \param o pitch shifting object as returned by new_aubio_pitchshift()
  \param pitchscale new pitch scale of the pitch shifting object

  pitchscale is a frequency ratio. It should be in the range [0.25, 4].

  \return 0 if successfull, non-zero otherwise

*/
uint_t aubio_pitchshift_set_pitchscale (aubio_pitchshift_t * o,
        smpl_t pitchscale);

/** get the pitchscale of the pitch shifting object

  \param o pitch shifting object as returned by ::new_aubio_pitchshift()

  \return pitchscale of the pitch shifting object

*/
smpl_t aubio_pitchshift_get_pitchscale (aubio_pitchshift_t * o);

/** set the transposition of the pitch shifting object, in semitones

  \param o pitch shifting object as returned by new_aubio_pitchshift()
  \param transpose new pitch transposition of the pitch shifting object,
  expressed in semitones (should be in the range [-24;+24])

  \return 0 if successfull, non-zero otherwise

*/
uint_t aubio_pitchshift_set_transpose (aubio_pitchshift_t * o,
        smpl_t transpose);

/** get the transposition of the pitch shifting object, in semitones

  \param o pitch shifting object as returned by ::new_aubio_pitchshift()

  \return transposition of the pitch shifting object, in semitones

*/
smpl_t aubio_pitchshift_get_transpose (aubio_pitchshift_t * o);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_PITCHSHIFT_H */

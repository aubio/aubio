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

#ifndef AUBIO_TIMESTRETCH_H
#define AUBIO_TIMESTRETCH_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  time stretching object

  ::aubio_timestretch_t can be used to open a source file, read samples from
  it, time-stretch them, and write out the modified samples.

  The time-stretching factor can be changed at any time using
  aubio_timestretch_set_stretch().

  A transposition can also be applied and changed at any time with
  aubio_timestretch_set_transpose().

  \example effects/test-timestretch.c

*/

/** time stretching object */
typedef struct _aubio_timestretch_t aubio_timestretch_t;

/** execute time stretching on an input signal frame

  \param o time stretching object as returned by new_aubio_timestretch()
  \param out timestretched output of size [hop_size]
  \param read number of frames actually wrote out

*/
void aubio_timestretch_do (aubio_timestretch_t * o, fvec_t * out,
   uint_t * read);

/** deletion of the time stretching object

  \param o time stretching object as returned by new_aubio_timestretch()

*/
void del_aubio_timestretch (aubio_timestretch_t * o);

/** creation of the time stretching object

  \param method time stretching algorithm ("default")
  \param stretch initial time stretching factor
  \param hop_size block size at which the frames should be produced
  \param samplerate sampling rate of the signal

  \return newly created ::aubio_timestretch_t

*/
aubio_timestretch_t *new_aubio_timestretch (const char_t * method,
    smpl_t stretch, uint_t hop_size, uint_t samplerate);

/** push length samples from in to time stretching object

  \param o time stretching object as returned by ::new_aubio_timestretch()
  \param in input vector of new samples to push to time stretching object
  \param length number of new samples to push from input vector

  \return number of currently available samples

 */
sint_t aubio_timestretch_push(aubio_timestretch_t * o, fvec_t *in,
    uint_t length);

/** get number of currently available samples from time stretching object

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return number of currently available samples

 */
sint_t aubio_timestretch_get_available(aubio_timestretch_t * o);

/** get the latency of the time stretching object, in samples

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return latency of the time stretching object in samples

*/
uint_t aubio_timestretch_get_latency (aubio_timestretch_t * o);

/** get the samplerate of the time stretching object

  Call after new_aubio_timestretch() was called with 0 to match the original
  samplerate of the input file.

  \param o time stretching object as returned by new_aubio_timestretch()

  \return samplerate of the time stretching object

 */
uint_t aubio_timestretch_get_samplerate (aubio_timestretch_t * o);

/** set the stretching ratio of the time stretching object

  \param o time stretching object as returned by new_aubio_timestretch()
  \param stretch new time stretching ratio of the time stretching object
  (should be in the range [0.025; 10.])

  \return 0 if successfull, non-zero otherwise

*/
uint_t aubio_timestretch_set_stretch (aubio_timestretch_t * o, smpl_t stretch);

/** get the transposition of the time stretching object, in semitones

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return time stretching ratio of the time stretching object, in the range
  [0.025; 10.]

*/
smpl_t aubio_timestretch_get_stretch (aubio_timestretch_t * o);

/** set the pitch scale of the time stretching object

  \param o time stretching object as returned by new_aubio_timestretch()
  \param pitchscale new pitch scale of the time stretching object

  pitchscale is a frequency ratio. It should be in the range [0.25, 4].

  \return 0 if successfull, non-zero otherwise

*/
uint_t aubio_timestretch_set_pitchscale (aubio_timestretch_t * o,
        smpl_t pitchscale);

/** get the pitchscale of the time stretching object

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return pitchscale of the time stretching object

*/
smpl_t aubio_timestretch_get_pitchscale (aubio_timestretch_t * o);

/** set the transposition of the time stretching object, in semitones

  \param o time stretching object as returned by new_aubio_timestretch()

  \param transpose new pitch transposition of the time stretching object,
  expressed in semitones (should be in the range [-24;+24])

  \return 0 if successfull, non-zero otherwise

*/
uint_t aubio_timestretch_set_transpose (aubio_timestretch_t * o,
        smpl_t transpose);

/** get the transposition of the time stretching object, in semitones

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return transposition of the time stretching object, in semitones

*/
smpl_t aubio_timestretch_get_transpose (aubio_timestretch_t * o);

/** reset the time stretching object

  \param o time stretching object as returned by ::new_aubio_timestretch()

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_timestretch_reset(aubio_timestretch_t * o);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_TIMESTRETCH_H */

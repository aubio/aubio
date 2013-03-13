/*
  Copyright (C) 2006-2009 Paul Brossier <piem@aubio.org>

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

/** \file 
  
  Tempo detection object

  This object stores all the memory required for tempo detection algorithm
  and returns the estimated beat locations.

  \example tempo/test-tempo.c

*/

#ifndef TEMPO_H
#define TEMPO_H

#ifdef __cplusplus
extern "C" {
#endif

/** tempo detection structure */
typedef struct _aubio_tempo_t aubio_tempo_t;

/** create tempo detection object */
aubio_tempo_t * new_aubio_tempo (char_t * method, 
    uint_t buf_size, uint_t hop_size, uint_t samplerate);

/** execute tempo detection */
void aubio_tempo_do (aubio_tempo_t *o, fvec_t * input, fvec_t * tempo);

/** set tempo detection silence threshold  */
uint_t aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence);

/** set tempo detection peak picking threshold  */
uint_t aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold);

/** get current tempo

  \param bt beat tracking object

  Returns the currently observed tempo, or 0 if no consistent value is found

*/
smpl_t aubio_tempo_get_bpm(aubio_tempo_t * bt);

/** get current tempo confidence

  \param bt beat tracking object

  Returns the confidence with which the tempo has been observed, 0 if no
  consistent value is found.

*/
smpl_t aubio_tempo_get_confidence(aubio_tempo_t * bt);

/** delete tempo detection object */
void del_aubio_tempo(aubio_tempo_t * o);

#ifdef __cplusplus
}
#endif

#endif /* TEMPO_H */

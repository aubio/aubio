/*
   Copyright (C) 2006 Paul Brossier

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
  
  Tempo detection driver

  This object stores all the memory required for tempo detection algorithm
  and returns the estimated beat locations.

*/

#ifndef TEMPO_H
#define TEMPO_H

#ifdef __cplusplus
extern "C" {
#endif

/** tempo detection structure */
typedef struct _aubio_tempo_t aubio_tempo_t;

/** create tempo detection object */
aubio_tempo_t * new_aubio_tempo (aubio_onsetdetection_type type_onset, 
    uint_t buf_size, uint_t hop_size, uint_t channels);

/** execute tempo detection */
void aubio_tempo(aubio_tempo_t *o, fvec_t * input, fvec_t * tempo);

/** set tempo detection silence threshold  */
void aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence);

/** set tempo detection peak picking threshold  */
void aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold);

/** delete tempo detection object */
void del_aubio_tempo(aubio_tempo_t * o);

#ifdef __cplusplus
}
#endif

#endif /* TEMPO_H */

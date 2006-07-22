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

#ifndef ONSET_H
#define ONSET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_onset_t aubio_onset_t;

/** create onset detection object */
aubio_onset_t * new_aubio_onset (aubio_onsetdetection_type type_onset, 
    uint_t buf_size, uint_t hop_size, uint_t channels);

/** execute onset detection */
void aubio_onset(aubio_onset_t *o, fvec_t * input, fvec_t * onset);

/** set onset detection silence threshold  */
void aubio_onset_set_silence(aubio_onset_t * o, smpl_t silence);

/** set onset detection peak picking threshold  */
void aubio_onset_set_threshold(aubio_onset_t * o, smpl_t threshold);

/** set onset detection peak picking threshold  */
void aubio_onset_set_minioi(aubio_onset_t * o, uint_t minioi);

/** delete onset detection object */
void del_aubio_onset(aubio_onset_t * o);

#ifdef __cplusplus
}
#endif

#endif /* ONSET_H */

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

typedef enum {
        aubio_pitch_yin,
        aubio_pitch_mcomb,
        aubio_pitch_schmitt,
        aubio_pitch_fcomb
} aubio_pitchdetection_type;

typedef enum {
        aubio_pitchm_freq,
        aubio_pitchm_midi,
        aubio_pitchm_cent,
        aubio_pitchm_bin
} aubio_pitchdetection_mode;

typedef struct _aubio_pitchdetection_t aubio_pitchdetection_t;
	
smpl_t aubio_pitchdetection(aubio_pitchdetection_t * p, fvec_t * ibuf);
smpl_t aubio_pitchdetection_mcomb(aubio_pitchdetection_t *p, fvec_t * ibuf);
smpl_t aubio_pitchdetection_yin(aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_schmitt(aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_fcomb(aubio_pitchdetection_t *p, fvec_t *ibuf);

void aubio_pitchdetection_set_yinthresh(aubio_pitchdetection_t *p, smpl_t thres);
void del_aubio_pitchdetection(aubio_pitchdetection_t * p);

aubio_pitchdetection_t * new_aubio_pitchdetection(uint_t bufsize, 
		uint_t hopsize, 
		uint_t channels,
		uint_t samplerate,
		aubio_pitchdetection_type type,
		aubio_pitchdetection_mode mode);

#ifdef __cplusplus
}
#endif

#endif /*PITCHDETECTION_H*/ 

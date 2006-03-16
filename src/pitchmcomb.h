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
 * spectral pitch detection function
 * 
 * \todo check/fix peak picking
 */

#ifndef PITCHMCOMB_H
#define PITCHMCOMB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_pitchmcomb_t aubio_pitchmcomb_t;

smpl_t aubio_pitchmcomb_detect(aubio_pitchmcomb_t * p, cvec_t * fftgrain);
uint_t aubio_pitch_cands(aubio_pitchmcomb_t * p, cvec_t * fftgrain, smpl_t * cands);
aubio_pitchmcomb_t * new_aubio_pitchmcomb(uint_t size, uint_t channels, uint_t samplerate);
void del_aubio_pitchmcomb(aubio_pitchmcomb_t *p);

#ifdef __cplusplus
}
#endif

#endif/*PITCHMCOMB_H*/ 

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

/** @file
 * Phase vocoder object
 */

#ifndef _PHASEVOC_H
#define _PHASEVOC_H

#ifdef __cplusplus
extern "C" {
#endif

/** phasevocoder object */
typedef struct _aubio_pvoc_t aubio_pvoc_t;

/** create phase vocoder object */
aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s, uint_t channels);
/** delete phase vocoder object */
void del_aubio_pvoc(aubio_pvoc_t *pv);

/** 
 * fill pvoc with inp[c][hop_s] 
 * slide current buffer 
 * calculate norm and phas of current grain 
 */
void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t *in, cvec_t * fftgrain);
/**
 * do additive resynthesis to 
 * from current norm and phase
 * to out[c][hop_s] 
 */
void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

/** get window size */
uint_t aubio_pvoc_get_win(aubio_pvoc_t* pv);
/** get hop size */
uint_t aubio_pvoc_get_hop(aubio_pvoc_t* pv);
/** get channel number */
uint_t aubio_pvoc_get_channels(aubio_pvoc_t* pv);

#ifdef __cplusplus
}
#endif 

#endif

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

#ifndef FILTER_H
#define FILTER_H

/** \file 
 * filter
 *
 * \f$ y[n] = b_1 x[n] + ... + b_{order} x[n-order] -
 *      a_2 y[n-1] - ... - a_{order} y[n-order]\f$
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_filter_t aubio_filter_t;
void aubio_filter_do(aubio_filter_t * b, fvec_t * in);
void aubio_filter_do_outplace(aubio_filter_t * b, fvec_t * in, fvec_t * out);
void aubio_filter_do_filtfilt(aubio_filter_t * b, fvec_t * in, fvec_t * tmp);
aubio_filter_t * new_aubio_filter(uint_t samplerate, uint_t order);
aubio_filter_t * new_aubio_adsgn_filter(uint_t samplerate);
aubio_filter_t * new_aubio_cdsgn_filter(uint_t samplerate);

#ifdef __cplusplus
}
#endif

#endif /*FILTER_H*/

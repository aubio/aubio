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

#ifndef BIQUAD_H
#define BIQUAD_H

/** \file 
 * biquad filter
 *
 * \f$ y[n] = b_1 x[n] + b_2 x[n-1] + b_3 x[n-2] -
 *      a_2 y[n-1] - a_3 y[n-2] \f$
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_biquad_t aubio_biquad_t;

void aubio_biquad_do(aubio_biquad_t * b, fvec_t * in);
void aubio_biquad_do_filtfilt(aubio_biquad_t * b, fvec_t * in, fvec_t * tmp);
aubio_biquad_t * new_aubio_biquad(lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);

#ifdef __cplusplus
}
#endif

#endif /*BIQUAD_H*/

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

  Second order Infinite Impulse Response filter

  This file implements a normalised biquad filter (second order IIR):
 
  \f$ y[n] = b_1 x[n] + b_2 x[n-1] + b_3 x[n-2] - a_2 y[n-1] - a_3 y[n-2] \f$

  The filtfilt version runs the filter twice, forward and backward, to
  compensate the phase shifting of the forward operation.

*/

#ifdef __cplusplus
extern "C" {
#endif

/** biquad filter object */
typedef struct _aubio_biquad_t aubio_biquad_t;

/** filter input vector

  \param b biquad object as returned by new_aubio_biquad
  \param in input vector to filter

*/
void aubio_biquad_do(aubio_biquad_t * b, fvec_t * in);
/** filter input vector forward and backward

  \param b biquad object as returned by new_aubio_biquad
  \param in input vector to filter
  \param tmp memory space to use for computation

*/
void aubio_biquad_do_filtfilt(aubio_biquad_t * b, fvec_t * in, fvec_t * tmp);
/** create new biquad filter

  \param b1 forward filter coefficient
  \param b2 forward filter coefficient
  \param b3 forward filter coefficient
  \param a2 feedback filter coefficient
  \param a3 feedback filter coefficient

*/
aubio_biquad_t * new_aubio_biquad(lsmp_t b1, lsmp_t b2, lsmp_t b3, lsmp_t a2, lsmp_t a3);

/** delete biquad filter 
 
  \param b biquad object to delete 

*/
void del_aubio_biquad(aubio_biquad_t * b);

#ifdef __cplusplus
}
#endif

#endif /*BIQUAD_H*/

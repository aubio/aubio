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

  Infinite Impulse Response filter

  This file implements IIR filters of any order:
 
  \f$ y[n] = b_1 x[n] + ... + b_{order} x[n-order] -
       a_2 y[n-1] - ... - a_{order} y[n-order]\f$

  The filtfilt version runs the filter twice, forward and backward, to
  compensate the phase shifting of the forward operation.

*/

#ifdef __cplusplus
extern "C" {
#endif

/** IIR filter object */
typedef struct _aubio_filter_t aubio_filter_t;

/** filter input vector (in-place)

  \param b biquad object as returned by new_aubio_biquad
  \param in input vector to filter

*/
void aubio_filter_do(aubio_filter_t * b, fvec_t * in);
/** filter input vector (out-of-place)

  \param b biquad object as returned by new_aubio_biquad
  \param in input vector to filter
  \param out output vector to store filtered input

*/
void aubio_filter_do_outplace(aubio_filter_t * b, fvec_t * in, fvec_t * out);
/** filter input vector forward and backward

  \param b biquad object as returned by new_aubio_biquad
  \param in input vector to filter
  \param tmp memory space to use for computation

*/
void aubio_filter_do_filtfilt(aubio_filter_t * b, fvec_t * in, fvec_t * tmp);
/** create new IIR filter

  \param samplerate signal sampling rate
  \param order order of the filter (number of coefficients)

*/
aubio_filter_t * new_aubio_filter(uint_t samplerate, uint_t order);
/** create a new A-design filter 

  \param samplerate sampling-rate of the signal to filter 

*/
aubio_filter_t * new_aubio_adsgn_filter(uint_t samplerate);
/** create a new C-design filter 

  \param samplerate sampling-rate of the signal to filter 

*/
aubio_filter_t * new_aubio_cdsgn_filter(uint_t samplerate);
/** delete a filter object
 
  \param f filter object to delete

*/
void del_aubio_filter(aubio_filter_t * f);

#ifdef __cplusplus
}
#endif

#endif /*FILTER_H*/

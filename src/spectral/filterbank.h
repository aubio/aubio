/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>
                      and Amaury Hazan <ahazan@iua.upf.edu>

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

  Filterbank object

  General-purpose spectral filterbank object.

  \example spectral/test-filterbank.c

*/

#ifndef FILTERBANK_H
#define FILTERBANK_H

#ifdef __cplusplus
extern "C"
{
#endif

/** filterbank object

  This object stores a matrix of spectral filter coefficients.

 */
typedef struct _aubio_filterbank_t aubio_filterbank_t;

/** create filterbank object

  \param n_filters number of filters to create
  \param win_s size of analysis buffer (and length the FFT transform)

*/
aubio_filterbank_t *new_aubio_filterbank (uint_t n_filters, uint_t win_s);

/** destroy filterbank object

  \param fb filterbank, as returned by new_aubio_filterbank() method

*/
void del_aubio_filterbank (aubio_filterbank_t * fb);

/** compute filterbank

  \param fb filterbank containing     nfilt x win_s filter coefficients
  \param in input spectrum containing chans x win_s spectrum
  \param out output vector containing chans x nfilt output values

*/
void aubio_filterbank_do (aubio_filterbank_t * fb, cvec_t * in, fvec_t * out);

/** return a pointer to the matrix object containing all filter coefficients 

  \param f filterbank object to get coefficients from

 */
fmat_t *aubio_filterbank_get_coeffs (aubio_filterbank_t * f);

/** copy filter coefficients to the filterbank

  \param f filterbank object to set coefficients
  \param filters filter bank coefficients to copy from

 */
uint_t aubio_filterbank_set_coeffs (aubio_filterbank_t * f, fmat_t * filters);

#ifdef __cplusplus
}
#endif

#endif                          // FILTERBANK_H

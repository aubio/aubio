/*
  Copyright (C) 2007-2013 Paul Brossier <piem@aubio.org>
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

#ifndef _AUBIO_FILTERBANK_H
#define _AUBIO_FILTERBANK_H

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

  \param f filterbank object, as returned by new_aubio_filterbank()

*/
void del_aubio_filterbank (aubio_filterbank_t * f);

/** compute filterbank

  \param f filterbank object, as returned by new_aubio_filterbank()
  \param in input spectrum containing an input spectrum of length `win_s`
  \param out output vector containing the energy found in each band, `nfilt` output values

*/
void aubio_filterbank_do (aubio_filterbank_t * f, cvec_t * in, fvec_t * out);

/** return a pointer to the matrix object containing all filter coefficients

  \param f filterbank object, as returned by new_aubio_filterbank()

 */
fmat_t *aubio_filterbank_get_coeffs (aubio_filterbank_t * f);

/** copy filter coefficients to the filterbank

  \param f filterbank object, as returned by new_aubio_filterbank()
  \param filters filter bank coefficients to copy from

 */
uint_t aubio_filterbank_set_coeffs (aubio_filterbank_t * f, fmat_t * filters);

#ifdef __cplusplus
}
#endif

#endif /* _AUBIO_FILTERBANK_H */

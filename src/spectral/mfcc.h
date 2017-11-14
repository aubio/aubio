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

  Mel-Frequency Cepstrum Coefficients object

  This object computes MFCC coefficients on an input cvec_t.

  The implementation follows the specifications established by Malcolm Slaney
  in its Auditory Toolbox, available online (see file mfcc.m).

  http://engineering.ecn.purdue.edu/~malcolm/interval/1998-010/

  \example spectral/test-mfcc.c

*/

#ifndef AUBIO_MFCC_H
#define AUBIO_MFCC_H

#ifdef __cplusplus
extern "C"
{
#endif

/** mfcc object */
typedef struct _aubio_mfcc_t aubio_mfcc_t;

/** create mfcc object

  \param buf_size size of analysis buffer (and length the FFT transform)
  \param samplerate audio sampling rate
  \param n_coeffs number of desired coefficients
  \param n_filters number of desired filters

*/
AUBIO_API aubio_mfcc_t *new_aubio_mfcc (uint_t buf_size,
    uint_t n_filters, uint_t n_coeffs, uint_t samplerate);

/** delete mfcc object

  \param mf mfcc object as returned by new_aubio_mfcc

*/
AUBIO_API void del_aubio_mfcc (aubio_mfcc_t * mf);

/** mfcc object processing

  \param mf mfcc object as returned by new_aubio_mfcc
  \param in input spectrum (buf_size long)
  \param out output mel coefficients buffer (n_coeffs long)

*/
AUBIO_API void aubio_mfcc_do (aubio_mfcc_t * mf, const cvec_t * in, fvec_t * out);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_MFCC_H */

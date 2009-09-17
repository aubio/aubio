/*
   Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>
                       and Amaury Hazan <ahazan@iua.upf.edu>

   This file is part of Aubio.

   Aubio is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Aubio is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with Aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef MFCC_H
#define MFCC_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct aubio_mfcc_t_ aubio_mfcc_t;

/** create mfcc object

  \param win_s size of analysis buffer (and length the FFT transform)
  \param samplerate audio sampling rate
  \param n_coefs number of desired coefficientss

*/
aubio_mfcc_t *new_aubio_mfcc (uint_t win_s, uint_t samplerate,
      uint_t n_filters, uint_t n_coefs);

/** delete mfcc object

  \param mf mfcc object as returned by new_aubio_mfcc

*/
void del_aubio_mfcc (aubio_mfcc_t * mf);

/** mfcc object processing

  \param mf mfcc object as returned by new_aubio_mfcc
  \param in input spectrum (win_s long)
  \param out output mel coefficients buffer (n_coeffs long)

*/
void aubio_mfcc_do (aubio_mfcc_t * mf, cvec_t * in, fvec_t * out);

#ifdef __cplusplus
}
#endif

#endif                          // MFCC_H

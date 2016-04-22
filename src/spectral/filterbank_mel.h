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

  Filterbank object coefficients initialization

  Functions to create set the ::aubio_filterbank_t coefficients to
    - ::aubio_filterbank_set_triangle_bands: overlapping triangular bands,
    - ::aubio_filterbank_set_mel_coeffs_slaney: Mel frequency bands.

  \example spectral/test-filterbank_mel.c

*/

#ifndef AUBIO_FILTERBANK_MEL_H
#define AUBIO_FILTERBANK_MEL_H

#ifdef __cplusplus
extern "C"
{
#endif

/** filterbank initialization with triangular and overlapping bands

  \param fb filterbank object
  \param freqs arbitrary array of boundary frequencies
  \param samplerate audio sampling rate

  This function computes the coefficients of the filterbank based on the
  boundaries found in freqs, in Hz, and using triangular overlapping bands.

*/
uint_t aubio_filterbank_set_triangle_bands (aubio_filterbank_t * fb,
    const fvec_t * freqs, smpl_t samplerate);

/** filterbank initialization for Mel filters using Slaney's coefficients

  \param fb filterbank object
  \param samplerate audio sampling rate

  The filter coefficients are built according to Malcolm Slaney's Auditory
  Toolbox, available at http://engineering.purdue.edu/~malcolm/interval/1998-010/
  (see file mfcc.m).

*/
uint_t aubio_filterbank_set_mel_coeffs_slaney (aubio_filterbank_t * fb,
    smpl_t samplerate);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_FILTERBANK_MEL_H */

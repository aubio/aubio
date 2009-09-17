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

/** \file

  Mel frequency filter bankd coefficients 

  Set filter bank coefficients to Mel frequency bands.

  The filter coefficients are built according to Malcolm Slaney's Auditory
  Toolbox available at http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/
  (see the file mfcc.m). 

*/

#ifndef FILTERBANK_MEL_H
#define FILTERBANK_MEL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "filterbank.h"

/** filterbank initialization for mel filters

  \param fb filterbank object
  \param freqs arbitrary array of boundary frequencies
  \param samplerate audio sampling rate

  This function computes the coefficients of the filterbank based on the
  boundaries found in freqs, in Hz, and using triangular overlapping windows.

*/
void aubio_filterbank_set_mel_coeffs (aubio_filterbank_t * fb,
    fvec_t * freqs, smpl_t samplerate);

/** filterbank initialization for Mel filters using Slaney's coefficients

  \param fb filterbank object
  \param samplerate audio sampling rate

  This function fills the filterbank coefficients according to Malcolm Slaney.

*/
void aubio_filterbank_set_mel_coeffs_slaney (aubio_filterbank_t * fb,
    smpl_t samplerate);

#ifdef __cplusplus
}
#endif

#endif                          // FILTERBANK_MEL_H

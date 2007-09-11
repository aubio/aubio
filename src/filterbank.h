/*
   Copyright (C) 2007 Amaury Hazan <ahazan@iua.upf.edu>
                  and Paul Brossier <piem@piem.org>

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

/** \file

  Filterbank object

  General-purpose spectral filterbank object. Comes with mel-filter initialization function.

*/

#ifndef FILTERBANK_H
#define FILTERBANK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aubio_filterbank_t_ aubio_filterbank_t;

/** create filterbank object

  \param win_s size of analysis buffer (and length the FFT transform)
  \param n_filters number of filters to create

*/

aubio_filterbank_t * new_aubio_filterbank(uint_t n_filters, uint_t win_s);

/** filterbank initialization for mel filters

  
  \param n_filters number of filters
  \param win_s window size
  \param samplerate
  \param freq_min lowest filter frequency
  \param freq_max highest filter frequency

*/
aubio_filterbank_t * new_aubio_filterbank_mfcc(uint_t n_filters, uint_t win_s, uint_t samplerate, smpl_t freq_min, smpl_t freq_max);

/** filterbank initialization for mel filters

  \param n_filters number of filters
  \param win_s window size
  \param samplerate
  \param freq_min lowest filter frequency
  \param freq_max highest filter frequency

*/
aubio_filterbank_t * new_aubio_filterbank_mfcc_2(uint_t n_filters, uint_t win_s, uint_t samplerate, smpl_t freq_min, smpl_t freq_max);


/** destroy filterbank object

  \param fb filterbank, as returned by new_aubio_filterbank method

*/
void del_aubio_filterbank(aubio_filterbank_t * fb);

/** compute filterbank

*/
void aubio_filterbank_do(aubio_filterbank_t * fb, cvec_t * in, fvec_t *out);

/** dump filterbank filter tables in a txt file

*/
void aubio_dump_filterbank(aubio_filterbank_t * fb);

#ifdef __cplusplus
}
#endif

#endif // FILTERBANK_H

/*
   Copyright (C) 2007 Amaury Hazan
   adapted to aubio from LibXtract
   http://libxtract.sourceforge.net/
   

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

#ifndef AUBIOFILTERBANK_H
#define AUBIOFILTERBANK_H

#ifdef __cplusplus
extern "C" {
#endif



typedef struct aubio_filterbank_t_ aubio_filterbank_t;

/** create filterbank object

  \param win_s size of analysis buffer (and length the FFT transform)
  \param n_filters number of filters to create

*/

aubio_filterbank_t * new_aubio_filterbank(uint_t n_filters, uint_t win_s);

/** destroy filterbank object

  \param fb filterbank, as returned by new_aubio_filterbank method

*/
void del_aubio_filterbank(aubio_filterbank_t * fb);

/** filterbank initialization for mel filters

  \param fb filterbank, as returned by new_aubio_filterbank method
  \param nyquist nyquist frequency, i.e. half of the sampling rate
  \param style libxtract style
  \param freqmin lowest filter frequency
  \param freqmax highest filter frequency

*/
void aubio_filterbank_mfcc_init(aubio_filterbank_t * fb, smpl_t nyquist, int style, smpl_t freq_min, smpl_t freq_max);

// Initialization

/** \brief A function to initialise a mel filter bank 
 * 
 * It is up to the caller to pass in a pointer to memory allocated for freq_bands arrays of length N. This function populates these arrays with magnitude coefficients representing the mel filterbank on a linear scale 
 */
int aubio_mfcc_init(int N, smpl_t nyquist, int style, smpl_t freq_min, smpl_t freq_max, int freq_bands, smpl_t ** fft_tables);

#ifdef __cplusplus
}
#endif

#endif

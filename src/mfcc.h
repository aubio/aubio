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

/* part of this mfcc implementation were inspired from LibXtract
   http://libxtract.sourceforge.net/
*/

#ifndef MFCC_H 
#define MFCC_H 

#ifdef __cplusplus
extern "C" {
#endif

#include "sample.h"
#include "filterbank.h"

typedef struct aubio_mfcc_t_ aubio_mfcc_t;

/** create mfcc object

  \param win_s size of analysis buffer (and length the FFT transform)
  \param samplerate 
  \param n_coefs: number of desired coefs
  \param lowfreq: lowest frequency to use in filterbank
  \param highfreq highest frequency to use in filterbank
  \param channels number of channels

*/
aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate, uint_t n_filters, uint_t n_coefs, smpl_t lowfreq, smpl_t highfreq, uint_t channels);
/** delete mfcc object

  \param mf mfcc object as returned by new_aubio_mfcc

*/
void del_aubio_mfcc(aubio_mfcc_t *mf);
/** mfcc object processing

  \param mf mfcc object as returned by new_aubio_mfcc
  \param in input spectrum (win_s long)
  \param out output mel coefficients buffer (n_filters/2 +1 long)

*/
void aubio_mfcc_do(aubio_mfcc_t * mf, cvec_t *in, fvec_t *out);

/** intermediate dct involved in aubio_mfcc_do

  \param mf mfcc object as returned by new_aubio_mfcc
  \param in input spectrum (n_filters long)
  \param out output mel coefficients buffer (n_filters/2 +1 long)

*/
void aubio_dct_do(aubio_mfcc_t * mf, fvec_t *in, fvec_t *out);

#ifdef __cplusplus
}
#endif

#endif // MFCC_H

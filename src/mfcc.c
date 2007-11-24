/*
   Copyright (C) 2006 Amaury Hazan
   Ported to aubio from LibXtract
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

#include "aubio_priv.h"
#include "sample.h"
#include "fft.h"
#include "filterbank.h"
#include "mfcc.h"
#include "math.h"

/** Internal structure for mfcc object **/

struct aubio_mfcc_t_{
  uint_t win_s;             /** grain length */
  uint_t samplerate;        /** sample rate (needed?) */
  uint_t channels;          /** number of channels */
  uint_t n_filters;         /** number of  *filters */
  uint_t n_coefs;           /** number of coefficients (<= n_filters/2 +1) */
  smpl_t lowfreq;           /** lowest frequency for filters */ 
  smpl_t highfreq;          /** highest frequency for filters */
  aubio_filterbank_t * fb;  /** filter bank */
  fvec_t * in_dct;          /** input buffer for dct * [fb->n_filters] */
  aubio_fft_t * fft_dct;   /** fft object for dct */
  cvec_t * fftgrain_dct;    /** output buffer for dct */
};


aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate, uint_t n_filters, uint_t n_coefs, smpl_t lowfreq, smpl_t highfreq, uint_t channels){
  /** allocating space for mfcc object */
  aubio_mfcc_t * mfcc = AUBIO_NEW(aubio_mfcc_t);

  //we need (n_coefs-1)*2 filters to obtain n_coefs coefficients after dct
  //uint_t n_filters = (n_coefs-1)*2;
  
  mfcc->win_s=win_s;
  mfcc->samplerate=samplerate;
  mfcc->channels=channels;
  mfcc->n_filters=n_filters;
  mfcc->n_coefs=n_coefs;
  mfcc->lowfreq=lowfreq;
  mfcc->highfreq=highfreq;

  
  /** filterbank allocation */
  mfcc->fb = new_aubio_filterbank_mfcc(n_filters, mfcc->win_s, samplerate, lowfreq, highfreq);

  /** allocating space for fft object (used for dct) */
  mfcc->fft_dct=new_aubio_fft(n_filters, 1);

  /** allocating buffers */
  mfcc->in_dct=new_fvec(mfcc->win_s, 1);
  
  mfcc->fftgrain_dct=new_cvec(n_filters, 1);

  return mfcc;
};

void del_aubio_mfcc(aubio_mfcc_t *mf){
  /** deleting filterbank */
  del_aubio_filterbank(mf->fb);
  /** deleting fft object */
  del_aubio_fft(mf->fft_dct);
  /** deleting buffers */
  del_fvec(mf->in_dct);
  del_cvec(mf->fftgrain_dct);
  
  /** deleting mfcc object */
  AUBIO_FREE(mf);
}


/** intermediate dct involved in aubio_mfcc_do

  \param mf mfcc object as returned by new_aubio_mfcc
  \param in input spectrum (n_filters long)
  \param out output mel coefficients buffer (n_filters/2 +1 long)

*/
void aubio_dct_do(aubio_mfcc_t * mf, fvec_t *in, fvec_t *out);

void aubio_mfcc_do(aubio_mfcc_t * mf, cvec_t *in, fvec_t *out){
    // compute filterbank
    aubio_filterbank_do(mf->fb, in, mf->in_dct);
    //TODO: check that zero padding 
    // the following line seems useless since the in_dct buffer has the correct size
    //for(n = filter + 1; n < N; n++) result[n] = 0; 
    
    aubio_dct_do(mf, mf->in_dct, out);

    return;
}

void aubio_dct_do(aubio_mfcc_t * mf, fvec_t *in, fvec_t *out){
    uint_t i;
    //compute mag spectrum
    aubio_fft_do (mf->fft_dct, in, mf->fftgrain_dct);
    //extract real part of fft grain
    for(i=0; i<mf->n_coefs ;i++){
    //for(i=0; i<out->length;i++){
      out->data[0][i]= mf->fftgrain_dct->norm[0][i]
        *COS(mf->fftgrain_dct->phas[0][i]);
    }
    return;
}


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

  /** grain length */
  uint_t win_s;
  
  /** sample rate (needed?) */
  uint_t samplerate;

  /** number of channels */
  uint_t channels;
  
  /** filter bank */
  aubio_filterbank_t * fb;

  /** number of coefficients (= fb->n_filters/2 +1) */
  uint_t n_coefs;

  /** lowest frequency for filters */ 
  smpl_t lowfreq;
  
  /** highest frequency for filters */
  smpl_t highfreq;

  /** input buffer for dct * [fb->n_filters] */
  fvec_t * in_dct;

  /** fft object for dct */
  aubio_mfft_t * fft_dct;

  /** output buffer for dct */
  cvec_t * fftgrain_dct;

};


aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate ,uint_t n_coefs, smpl_t lowfreq, smpl_t highfreq, uint_t channels){


  /** allocating space for mfcc object */
  
  aubio_mfcc_t * mfcc = AUBIO_NEW(aubio_mfcc_t);
  
  mfcc->win_s=win_s;
  mfcc->samplerate=samplerate;
  mfcc->channels=channels;
  mfcc->n_coefs=n_coefs;
  mfcc->lowfreq=lowfreq;
  mfcc->highfreq=highfreq;

  /** filterbank allocation */
  //we need (n_coefs-1)*2 filters to obtain n_coefs coefficients after dct
  mfcc->fb=new_aubio_filterbank((n_coefs-1)*2, mfcc->win_s);

  /** allocating space for fft object (used for dct) */
  mfcc->fft_dct=new_aubio_mfft(mfcc->win_s, 1);

  /** allocating buffers */
  
  mfcc->in_dct=new_fvec(mfcc->win_s, 1);
  
  mfcc->fftgrain_dct=new_cvec(mfcc->fb->n_filters, 1);

  /** populating the filterbank */
  
  aubio_filterbank_mfcc_init(mfcc->fb, (mfcc->samplerate)/2, XTRACT_EQUAL_GAIN, mfcc->lowfreq, mfcc->highfreq);

  return mfcc;

};


void del_aubio_mfcc(aubio_mfcc_t *mf){
  
  /** deleting filterbank */
  del_aubio_filterbank(mf->fb);
  /** deleting mfft object */
  del_aubio_mfft(mf->fft_dct);
  /** deleting buffers */
  del_fvec(mf->in_dct);
  del_cvec(mf->fftgrain_dct);
  
  /** deleting mfcc object */
  AUBIO_FREE(mf);

}


// Computation

void aubio_mfcc_do(aubio_mfcc_t * mf, cvec_t *in, fvec_t *out){

    aubio_filterbank_t *f = mf->fb;
    uint_t n, filter_cnt;

    for(filter_cnt = 0; filter_cnt < f->n_filters; filter_cnt++){
        mf->in_dct->data[0][filter_cnt] = 0.f;
        for(n = 0; n < mf->win_s; n++){
            mf->in_dct->data[0][filter_cnt] += in->norm[0][n] * f->filters[filter_cnt]->data[0][n];
        }
        mf->in_dct->data[0][filter_cnt] = LOG(mf->in_dct->data[0][filter_cnt] < XTRACT_LOG_LIMIT ? XTRACT_LOG_LIMIT : mf->in_dct->data[0][filter_cnt]);
    }

    //TODO: check that zero padding 
    // the following line seems useless since the in_dct buffer has the correct size
    //for(n = filter + 1; n < N; n++) result[n] = 0; 
    
    aubio_dct_do(mf, mf->in_dct, out);
    
    //return XTRACT_SUCCESS;
}

void aubio_dct_do(aubio_mfcc_t * mf, fvec_t *in, fvec_t *out){
    
    
    
    //fvec_t * momo = new_fvec(20, 1);
    //momo->data = data;
    
    //compute mag spectrum
    aubio_mfft_do (mf->fft_dct, in, mf->fftgrain_dct);

    int i;
    //extract real part of fft grain
    for(i=0; i<mf->n_coefs ;i++){
      out->data[0][i]= mf->fftgrain_dct->norm[0][i]*COS(mf->fftgrain_dct->phas[0][i]);
    }


    //return XTRACT_SUCCESS;
}


///////// OLD CODE

// int aubio_mfcc_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t * fft_dct, cvec_t * fftgrain_dct){
// 
//     aubio_mel_filter *f;
//     uint_t n, filter;
// 
//     f = (aubio_mel_filter *)argv;
//     printf("%d",f->n_filters);
// 
//     for(filter = 0; filter < f->n_filters; filter++){
//         result[filter] = 0.f;
//         for(n = 0; n < N; n++){
//             result[filter] += data[n] * f->filters[filter][n];
//         }
//         result[filter] = LOG(result[filter] < XTRACT_LOG_LIMIT ? XTRACT_LOG_LIMIT : result[filter]);
//     }
// 
//     //TODO: check that zero padding 
//     for(n = filter + 1; n < N; n++) result[n] = 0; 
//     
//     aubio_dct_do(result, f->n_filters, NULL, result, fft_dct, fftgrain_dct);
//     
//     return XTRACT_SUCCESS;
// }

// Added last two arguments to be able to pass from example

// int aubio_dct_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t * fft_dct, cvec_t * fftgrain_dct){
//     
//     
//     //call aubio p_voc in dct setting
// 
//     //TODO: fvec as input? Remove data length, N?
// 
//     fvec_t * momo = new_fvec(20, 1);
//     momo->data = data;
//     
//     //compute mag spectrum
//     aubio_mfft_do (fft_dct, data, fftgrain_dct);
// 
//     int i;
//     //extract real part of fft grain
//     for(i=0; i<N ;i++){
//       result[i]= fftgrain_dct->norm[0][i]*COS(fftgrain_dct->phas[0][i]);
//     }
// 
// 
//     return XTRACT_SUCCESS;
// }

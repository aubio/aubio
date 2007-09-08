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

#define VERY_SMALL_NUMBER 2e-42
#define USE_EQUAL_GAIN 1


/** Internal structure for mfcc object **/

struct aubio_mfcc_t_{
  uint_t win_s;             /** grain length */
  uint_t samplerate;        /** sample rate (needed?) */
  uint_t channels;          /** number of channels */
  uint_t n_coefs;           /** number of coefficients (= fb->n_filters/2 +1) */
  smpl_t lowfreq;           /** lowest frequency for filters */ 
  smpl_t highfreq;          /** highest frequency for filters */
  aubio_filterbank_t * fb;  /** filter bank */
  fvec_t * in_dct;          /** input buffer for dct * [fb->n_filters] */
  aubio_mfft_t * fft_dct;   /** fft object for dct */
  cvec_t * fftgrain_dct;    /** output buffer for dct */
};


/** filterbank initialization for mel filters

  \param fb filterbank, as returned by new_aubio_filterbank method
  \param nyquist nyquist frequency, i.e. half of the sampling rate
  \param style libxtract style
  \param freqmin lowest filter frequency
  \param freqmax highest filter frequency

*/
void aubio_filterbank_mfcc_init(aubio_filterbank_t * fb, smpl_t nyquist, int style, smpl_t freq_min, smpl_t freq_max);

aubio_mfcc_t * new_aubio_mfcc (uint_t win_s, uint_t samplerate ,uint_t n_coefs, smpl_t lowfreq, smpl_t highfreq, uint_t channels){
  /** allocating space for mfcc object */
  aubio_mfcc_t * mfcc = AUBIO_NEW(aubio_mfcc_t);

  //we need (n_coefs-1)*2 filters to obtain n_coefs coefficients after dct
  uint_t n_filters = (n_coefs-1)*2;
  
  mfcc->win_s=win_s;
  mfcc->samplerate=samplerate;
  mfcc->channels=channels;
  mfcc->n_coefs=n_coefs;
  mfcc->lowfreq=lowfreq;
  mfcc->highfreq=highfreq;

  /** filterbank allocation */
  mfcc->fb = new_aubio_filterbank(n_filters, mfcc->win_s);

  /** allocating space for fft object (used for dct) */
  mfcc->fft_dct=new_aubio_mfft(mfcc->win_s, 1);

  /** allocating buffers */
  mfcc->in_dct=new_fvec(mfcc->win_s, 1);
  
  mfcc->fftgrain_dct=new_cvec(n_filters, 1);

  /** populating the filterbank */
  aubio_filterbank_mfcc_init(mfcc->fb, (mfcc->samplerate)/2, mfcc->lowfreq, mfcc->highfreq);

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

void aubio_mfcc_do(aubio_mfcc_t * mf, cvec_t *in, fvec_t *out){

    aubio_filterbank_t *f = mf->fb;
    uint_t n, filter_cnt;

    for(filter_cnt = 0; filter_cnt < f->n_filters; filter_cnt++){
        mf->in_dct->data[0][filter_cnt] = 0.f;
        for(n = 0; n < mf->win_s; n++){
            mf->in_dct->data[0][filter_cnt] += in->norm[0][n] * f->filters[filter_cnt]->data[0][n];
        }
        mf->in_dct->data[0][filter_cnt] = LOG(mf->in_dct->data[0][filter_cnt] < VERY_SMALL_NUMBER ? VERY_SMALL_NUMBER : mf->in_dct->data[0][filter_cnt]);
    }

    //TODO: check that zero padding 
    // the following line seems useless since the in_dct buffer has the correct size
    //for(n = filter + 1; n < N; n++) result[n] = 0; 
    
    aubio_dct_do(mf, mf->in_dct, out);

    return;
}

void aubio_dct_do(aubio_mfcc_t * mf, fvec_t *in, fvec_t *out){
    //compute mag spectrum
    aubio_mfft_do (mf->fft_dct, in, mf->fftgrain_dct);

    int i;
    //extract real part of fft grain
    for(i=0; i<mf->n_coefs ;i++){
      out->data[0][i]= mf->fftgrain_dct->norm[0][i]*COS(mf->fftgrain_dct->phas[0][i]);
    }

    return;
}

void aubio_filterbank_mfcc_init(aubio_filterbank_t * fb, smpl_t nyquist, int style, smpl_t freq_min, smpl_t freq_max){

  int n, i, k, *fft_peak, M, next_peak; 
  smpl_t norm, mel_freq_max, mel_freq_min, norm_fact, height, inc, val, 
         freq_bw_mel, *mel_peak, *height_norm, *lin_peak;

  mel_peak = height_norm = lin_peak = NULL;
  fft_peak = NULL;
  norm = 1; 

  mel_freq_max = 1127 * log(1 + freq_max / 700);
  mel_freq_min = 1127 * log(1 + freq_min / 700);
  freq_bw_mel = (mel_freq_max - mel_freq_min) / fb->n_filters;

  mel_peak = (smpl_t *)malloc((fb->n_filters + 2) * sizeof(smpl_t)); 
  /* +2 for zeros at start and end */
  lin_peak = (smpl_t *)malloc((fb->n_filters + 2) * sizeof(smpl_t));
  fft_peak = (int *)malloc((fb->n_filters + 2) * sizeof(int));
  height_norm = (smpl_t *)malloc(fb->n_filters * sizeof(smpl_t));

  if(mel_peak == NULL || height_norm == NULL || 
      lin_peak == NULL || fft_peak == NULL)
    return NULL;

  M = fb->win_s >> 1;

  mel_peak[0] = mel_freq_min;
  lin_peak[0] = 700 * (exp(mel_peak[0] / 1127) - 1);
  fft_peak[0] = lin_peak[0] / nyquist * M;


  for (n = 1; n <= fb->n_filters; n++){  
    /*roll out peak locations - mel, linear and linear on fft window scale */
    mel_peak[n] = mel_peak[n - 1] + freq_bw_mel;
    lin_peak[n] = 700 * (exp(mel_peak[n] / 1127) -1);
    fft_peak[n] = lin_peak[n] / nyquist * M;
  }

  for (n = 0; n < fb->n_filters; n++){
    /*roll out normalised gain of each peak*/
    if (style == USE_EQUAL_GAIN){
      height = 1; 
      norm_fact = norm;
    }
    else{
      height = 2 / (lin_peak[n + 2] - lin_peak[n]);
      norm_fact = norm / (2 / (lin_peak[2] - lin_peak[0]));
    }
    height_norm[n] = height * norm_fact;
  }

  i = 0;

  for(n = 0; n < fb->n_filters; n++){

    /*calculate the rise increment*/
    if(n > 0)
      inc = height_norm[n] / (fft_peak[n] - fft_peak[n - 1]);
    else
      inc = height_norm[n] / fft_peak[n];
    val = 0;  

    /*zero the start of the array*/
    for(k = 0; k < i; k++)
      //fft_tables[n][k] = 0.f;
      fb->filters[n]->data[0][k]=0.f;

    /*fill in the rise */
    for(; i <= fft_peak[n]; i++){ 
      // fft_tables[n][i] = val;
      fb->filters[n]->data[0][k]=val;
      val += inc;
    }

    /*calculate the fall increment */
    inc = height_norm[n] / (fft_peak[n + 1] - fft_peak[n]);

    val = 0;
    next_peak = fft_peak[n + 1];

    /*reverse fill the 'fall' */
    for(i = next_peak; i > fft_peak[n]; i--){ 
      //fft_tables[n][i] = val;
      fb->filters[n]->data[0][k]=val;
      val += inc;
    }

    /*zero the rest of the array*/
    for(k = next_peak + 1; k < fb->win_s; k++)
      //fft_tables[n][k] = 0.f;
      fb->filters[n]->data[0][k]=0.f;
  }

  free(mel_peak);
  free(lin_peak);
  free(height_norm);
  free(fft_peak);

}


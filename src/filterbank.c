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

#include "aubio_priv.h"
#include "sample.h"
#include "filterbank.h"

#include "stdio.h"

#define USE_EQUAL_GAIN 1
#define VERY_SMALL_NUMBER 2e-42

/** \brief A structure to store a set of n_filters filters of lenghts win_s */
struct aubio_filterbank_t_ {
    uint_t win_s;
    uint_t n_filters;
    fvec_t **filters;
};

aubio_filterbank_t * new_aubio_filterbank(uint_t n_filters, uint_t win_s){
  /** allocating space for filterbank object */
  aubio_filterbank_t * fb = AUBIO_NEW(aubio_filterbank_t);
  uint_t filter_cnt;
  fb->win_s=win_s;
  fb->n_filters=n_filters;

  /** allocating filter tables */
  fb->filters=AUBIO_ARRAY(fvec_t*,n_filters);
  for (filter_cnt=0; filter_cnt<n_filters; filter_cnt++)
    /* considering one-channel filters */
    fb->filters[filter_cnt]=new_fvec(win_s, 1);

  return fb;
}

aubio_filterbank_t * new_aubio_filterbank_mfcc(uint_t n_filters, uint_t win_s, smpl_t samplerate, smpl_t freq_min, smpl_t freq_max){
  
  uint_t writelog=1;

  FILE * mlog;
  if(writelog==1) mlog=fopen("filterbank.txt","w");
  

  smpl_t nyquist = samplerate/2.;
  uint_t style = 1;
  aubio_filterbank_t * fb = new_aubio_filterbank(n_filters, win_s);

  uint_t n, i, k, *fft_peak, M, next_peak; 
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
  fft_peak = (uint_t *)malloc((fb->n_filters + 2) * sizeof(uint_t));
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
      fb->filters[n]->data[0][k]=0.f;

    /*fill in the rise */
    for(; i <= fft_peak[n]; i++){ 
      fb->filters[n]->data[0][k]=val;
      val += inc;
    }

    /*calculate the fall increment */
    inc = height_norm[n] / (fft_peak[n + 1] - fft_peak[n]);

    val = 0;
    next_peak = fft_peak[n + 1];

    /*reverse fill the 'fall' */
    for(i = next_peak; i > fft_peak[n]; i--){ 
      fb->filters[n]->data[0][k]=val;
      val += inc;
    }

    /*zero the rest of the array*/
    for(k = next_peak + 1; k < fb->win_s; k++)
      fb->filters[n]->data[0][k]=0.f;

    if(writelog){
      //dumping filter values
      smpl_t area_tmp=0.f;
      for(k = 0; k < fb->win_s; k++){
        fprintf(mlog,"%f ",fb->filters[n]->data[0][k]);
      }
      fprintf(mlog,"\n");
    }

  }

  free(mel_peak);
  free(lin_peak);
  free(height_norm);
  free(fft_peak);

  if(mlog) fclose(mlog);

  return fb;

}


void del_aubio_filterbank(aubio_filterbank_t * fb){
  uint_t filter_cnt;
  /** deleting filter tables first */
  for (filter_cnt=0; filter_cnt<fb->n_filters; filter_cnt++)
    del_fvec(fb->filters[filter_cnt]);
  AUBIO_FREE(fb->filters);
  AUBIO_FREE(fb);
}

void aubio_filterbank_do(aubio_filterbank_t * f, cvec_t * in, fvec_t *out) {
  uint_t n, filter_cnt;
  for(filter_cnt = 0; (filter_cnt < f->n_filters)
    && (filter_cnt < out->length); filter_cnt++){
      out->data[0][filter_cnt] = 0.f;
      for(n = 0; n < in->length; n++){
          out->data[0][filter_cnt] += in->norm[0][n] 
            * f->filters[filter_cnt]->data[0][n];
      }
      out->data[0][filter_cnt] =
        LOG(out->data[0][filter_cnt] < VERY_SMALL_NUMBER ? 
            VERY_SMALL_NUMBER : out->data[0][filter_cnt]);
  }

  return;
}

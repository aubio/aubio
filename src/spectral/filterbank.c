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


#include "aubio_priv.h"
#include "sample.h"
#include "spectral/filterbank.h"
#include "mathutils.h"

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

/*
FB initialization based on Slaney's auditory toolbox
TODO:
  *solve memory leak problems while
  *solve quantization issues when constructing signal:
    *bug for win_s=512
    *corrections for win_s=1024 -> why even filters with smaller amplitude

*/

aubio_filterbank_t * new_aubio_filterbank_mfcc(uint_t n_filters, uint_t win_s, uint_t samplerate, smpl_t freq_min, smpl_t freq_max){
  
  aubio_filterbank_t * fb = new_aubio_filterbank(n_filters, win_s);
  
  
  //slaney params
  smpl_t lowestFrequency = 133.3333;
  smpl_t linearSpacing = 66.66666666;
  smpl_t logSpacing = 1.0711703;

  uint_t linearFilters = 13;
  uint_t logFilters = 27;
  uint_t allFilters = linearFilters + logFilters;
  
  //buffers for computing filter frequencies
  fvec_t * freqs=new_fvec(allFilters+2 , 1);
  
  fvec_t * lower_freqs=new_fvec( allFilters, 1);
  fvec_t * upper_freqs=new_fvec( allFilters, 1);
  fvec_t * center_freqs=new_fvec( allFilters, 1);

  fvec_t * triangle_heights=new_fvec( allFilters, 1);
  //lookup table of each bin frequency in hz
  fvec_t * fft_freqs=new_fvec(win_s, 1);

  uint_t filter_cnt, bin_cnt;
  
  //first step: filling all the linear filter frequencies
  for(filter_cnt=0; filter_cnt<linearFilters; filter_cnt++){
    freqs->data[0][filter_cnt]=lowestFrequency+ filter_cnt*linearSpacing;
  }
  smpl_t lastlinearCF=freqs->data[0][filter_cnt-1];
  
  //second step: filling all the log filter frequencies
  for(filter_cnt=0; filter_cnt<logFilters+2; filter_cnt++){
    freqs->data[0][filter_cnt+linearFilters] = 
      lastlinearCF*(pow(logSpacing,filter_cnt+1));
  }

  //Option 1. copying interesting values to lower_freqs, center_freqs and upper freqs arrays
  //TODO: would be nicer to have a reference to freqs->data, anyway we do not care in this init step
    
  for(filter_cnt=0; filter_cnt<allFilters; filter_cnt++){
    lower_freqs->data[0][filter_cnt]=freqs->data[0][filter_cnt];
    center_freqs->data[0][filter_cnt]=freqs->data[0][filter_cnt+1];
    upper_freqs->data[0][filter_cnt]=freqs->data[0][filter_cnt+2];
  }

  //computing triangle heights so that each triangle has unit area
  for(filter_cnt=0; filter_cnt<allFilters; filter_cnt++){
    triangle_heights->data[0][filter_cnt] = 2./(upper_freqs->data[0][filter_cnt] 
      - lower_freqs->data[0][filter_cnt]);
  }
  
  //AUBIO_DBG("filter tables frequencies\n");
  //for(filter_cnt=0; filter_cnt<allFilters; filter_cnt++)
  //  AUBIO_DBG("filter n. %d %f %f %f %f\n",
  //    filter_cnt, lower_freqs->data[0][filter_cnt], 
  //    center_freqs->data[0][filter_cnt], upper_freqs->data[0][filter_cnt], 
  //    triangle_heights->data[0][filter_cnt]);

  //filling the fft_freqs lookup table, which assigns the frequency in hz to each bin
  for(bin_cnt=0; bin_cnt<win_s; bin_cnt++){
    fft_freqs->data[0][bin_cnt]= aubio_bintofreq(bin_cnt, samplerate, win_s);
  }

  //building each filter table
  for(filter_cnt=0; filter_cnt<allFilters; filter_cnt++){

    //TODO:check special case : lower freq =0
    //calculating rise increment in mag/Hz
    smpl_t riseInc= triangle_heights->data[0][filter_cnt]/(center_freqs->data[0][filter_cnt]-lower_freqs->data[0][filter_cnt]);
    
    //zeroing begining of filter
    for(bin_cnt=0; bin_cnt<win_s-1; bin_cnt++){
      fb->filters[filter_cnt]->data[0][bin_cnt]=0.f;
      if( fft_freqs->data[0][bin_cnt]  <= lower_freqs->data[0][filter_cnt] &&
          fft_freqs->data[0][bin_cnt+1] > lower_freqs->data[0][filter_cnt]) {
        break;
      }
    }
    bin_cnt++;
    
    //positive slope
    for(; bin_cnt<win_s-1; bin_cnt++){
      fb->filters[filter_cnt]->data[0][bin_cnt]=(fft_freqs->data[0][bin_cnt]-lower_freqs->data[0][filter_cnt])*riseInc;
      //if(fft_freqs->data[0][bin_cnt]<= center_freqs->data[0][filter_cnt] && fft_freqs->data[0][bin_cnt+1]> center_freqs->data[0][filter_cnt])
      if(fft_freqs->data[0][bin_cnt+1]> center_freqs->data[0][filter_cnt])
        break;
    }
    //bin_cnt++;
    
    //negative slope
    for(; bin_cnt<win_s-1; bin_cnt++){
      
      //checking whether last value is less than 0...
      smpl_t val=triangle_heights->data[0][filter_cnt]-(fft_freqs->data[0][bin_cnt]-center_freqs->data[0][filter_cnt])*riseInc;
      if(val>=0)
        fb->filters[filter_cnt]->data[0][bin_cnt]=val;
      else fb->filters[filter_cnt]->data[0][bin_cnt]=0.f;
      
      //if(fft_freqs->data[0][bin_cnt]<= upper_freqs->data[0][bin_cnt] && fft_freqs->data[0][bin_cnt+1]> upper_freqs->data[0][filter_cnt])
      //TODO: CHECK whether bugfix correct
      if(fft_freqs->data[0][bin_cnt+1]> upper_freqs->data[0][filter_cnt])
        break;
    }
    //bin_cnt++;
    
    //zeroing tail
    for(; bin_cnt<win_s; bin_cnt++)
      fb->filters[filter_cnt]->data[0][bin_cnt]=0.f;

  }
  
  
  del_fvec(freqs);
  del_fvec(lower_freqs);
  del_fvec(upper_freqs);
  del_fvec(center_freqs);

  del_fvec(triangle_heights);
  del_fvec(fft_freqs);

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

fvec_t * aubio_filterbank_getchannel(aubio_filterbank_t * f, uint_t channel) {
  if ( (channel < f->n_filters) ) { return f->filters[channel]; }
  else { return NULL; }
}

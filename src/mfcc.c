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
#include "mfcc.h"
#include "math.h"

/*
new_aubio_mfcc
aubio_mfcc_do
del_aubio_mfcc
*/

// Computation
// Added last two arguments to be able to pass from example



int aubio_mfcc_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t * fft_dct, cvec_t * fftgrain_dct){

    aubio_mel_filter *f;
    int n, filter;

    f = (aubio_mel_filter *)argv;
    
    for(filter = 0; filter < f->n_filters; filter++){
        result[filter] = 0.f;
        for(n = 0; n < N; n++){
            result[filter] += data[n] * f->filters[filter][n];
        }
        result[filter] = LOG(result[filter] < XTRACT_LOG_LIMIT ? XTRACT_LOG_LIMIT : result[filter]);
    }

    //TODO: check that zero padding 
    for(n = filter + 1; n < N; n++) result[n] = 0; 
    
    aubio_dct_do(result, f->n_filters, NULL, result, fft_dct, fftgrain_dct);
    
    return XTRACT_SUCCESS;
}

// Added last two arguments to be able to pass from example

int aubio_dct_do(const float *data, const int N, const void *argv, float *result, aubio_mfft_t * fft_dct, cvec_t * fftgrain_dct){
    
    
    //call aubio p_voc in dct setting

    //TODO: fvec as input? Remove data length, N?

    fvec_t * momo = new_fvec(20, 1);
    momo->data = data;
    
    //compute mag spectrum
    aubio_mfft_do (fft_dct, data, fftgrain_dct);

    int i;
    //extract real part of fft grain
    for(i=0; i<N ;i++){
      result[i]= fftgrain_dct->norm[0][i]*COS(fftgrain_dct->phas[0][i]);
    }


    return XTRACT_SUCCESS;
}

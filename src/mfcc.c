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


#include "mffc.h"
#include "aubiofilterbank.h"

// Computation

int aubio_mfcc_do(const float *data, const int N, const void *argv, float *result){

    aubio_mel_filter *f;
    int n, filter;

    f = (aubio_mel_filter *)argv;
    
    for(filter = 0; filter < f->n_filters; filter++){
        result[filter] = 0.f;
        for(n = 0; n < N; n++){
            result[filter] += data[n] * f->filters[filter][n];
        }
        result[filter] = log(result[filter] < XTRACT_LOG_LIMIT ? XTRACT_LOG_LIMIT : result[filter]);
    }

    for(n = filter + 1; n < N; n++) result[n] = 0; 
    
    aubio_dct_do(result, f->n_filters, NULL, result);
    
    return XTRACT_SUCCESS;
}

int aubio_dct_do(const float *data, const int N, const void *argv, float *result){
    
    fftwf_plan plan;
    
    plan = 
        fftwf_plan_r2r_1d(N, (float *) data, result, FFTW_REDFT00, FFTW_ESTIMATE);
    
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    return XTRACT_SUCCESS;
}
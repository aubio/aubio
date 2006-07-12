/*
   Copyright (C) 2003 Paul Brossier

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
#include "mathutils.h"
#include "fft.h"

#if FFTW3F_SUPPORT
#define fftw_malloc 		fftwf_malloc
#define fftw_free 		fftwf_free
#define fftw_execute 		fftwf_execute
#define fftw_plan_dft_r2c_1d 	fftwf_plan_dft_r2c_1d
#define fftw_plan_dft_c2r_1d 	fftwf_plan_dft_c2r_1d
#define fftw_plan		fftwf_plan
#define fftw_destroy_plan	fftwf_destroy_plan
#endif

#if FFTW3F_SUPPORT
#define real_t smpl_t
#else
#define real_t lsmp_t
#endif

struct _aubio_fft_t {
	uint_t fft_size;
	uint_t channels;
	real_t  	*in, *out;
	fft_data_t 	*specdata;
	fftw_plan 	pfw, pbw;
};

aubio_fft_t * new_aubio_fft(uint_t size) {
	aubio_fft_t * s = AUBIO_NEW(aubio_fft_t);
	/* allocate memory */
	s->in       = AUBIO_ARRAY(real_t,size);
	s->out      = AUBIO_ARRAY(real_t,size);
	s->specdata = (fft_data_t*)fftw_malloc(sizeof(fft_data_t)*size);
	/* create plans */
	s->pfw = fftw_plan_dft_r2c_1d(size, s->in,  s->specdata, FFTW_ESTIMATE);
	s->pbw = fftw_plan_dft_c2r_1d(size, s->specdata, s->out, FFTW_ESTIMATE);
	return s;
}

void del_aubio_fft(aubio_fft_t * s) {
	/* destroy data */
	fftw_destroy_plan(s->pfw);
	fftw_destroy_plan(s->pbw);
	fftw_free(s->specdata);
	AUBIO_FREE(s->out);
	AUBIO_FREE(s->in );
	AUBIO_FREE(s);
}

void aubio_fft_do(const aubio_fft_t * s, 
		const smpl_t * data, fft_data_t * spectrum, 
		const uint_t size) {
	uint_t i;
	for (i=0;i<size;i++) s->in[i] = data[i];
	fftw_execute(s->pfw);
	for (i=0;i<size;i++) spectrum[i] = s->specdata[i];
}

void aubio_fft_rdo(const aubio_fft_t * s, 
		const fft_data_t * spectrum, 
		smpl_t * data, 
		const uint_t size) {
	uint_t i;
	const smpl_t renorm = 1./(smpl_t)size;
	for (i=0;i<size;i++) s->specdata[i] = spectrum[i];
	fftw_execute(s->pbw);
	for (i=0;i<size;i++) data[i] = s->out[i]*renorm;
}


void aubio_fft_getnorm(smpl_t * norm, fft_data_t * spectrum, uint_t size) {
	uint_t i;
	for (i=0;i<size;i++) norm[i] = ABSC(spectrum[i]);
}

void aubio_fft_getphas(smpl_t * phas, fft_data_t * spectrum, uint_t size) {
	uint_t i;
	for (i=0;i<size;i++) phas[i] = ARGC(spectrum[i]);
}


/* new interface aubio_mfft */
struct _aubio_mfft_t {
        aubio_fft_t * fft;      /* fftw interface */
        fft_data_t ** spec;     /* complex spectral data */
        uint_t winsize;
        uint_t channels;
};

aubio_mfft_t * new_aubio_mfft(uint_t winsize, uint_t channels){
        uint_t i;
	aubio_mfft_t * fft = AUBIO_NEW(aubio_mfft_t);
	fft->winsize       = winsize;
	fft->channels      = channels;
	fft->fft           = new_aubio_fft(winsize);
	fft->spec          = AUBIO_ARRAY(fft_data_t*,channels);
        for (i=0; i < channels; i++)
                fft->spec[i] = AUBIO_ARRAY(fft_data_t,winsize);
        return fft;
}

/* execute stft */
void aubio_mfft_do (aubio_mfft_t * fft,fvec_t * in,cvec_t * fftgrain){
        uint_t i=0;
        /* execute stft */
        for (i=0; i < fft->channels; i++) {
                aubio_fft_do (fft->fft,in->data[i],fft->spec[i],fft->winsize);
                /* put norm and phase into fftgrain */
                aubio_fft_getnorm(fftgrain->norm[i], fft->spec[i], fft->winsize/2+1);
                aubio_fft_getphas(fftgrain->phas[i], fft->spec[i], fft->winsize/2+1);
        }
}

/* execute inverse fourier transform */
void aubio_mfft_rdo(aubio_mfft_t * fft,cvec_t * fftgrain, fvec_t * out){
        uint_t i=0,j;
        for (i=0; i < fft->channels; i++) {
                for (j=0; j<fft->winsize/2+1; j++) {
                        fft->spec[i][j]  = CEXPC(I*aubio_unwrap2pi(fftgrain->phas[i][j]));
                        fft->spec[i][j] *= fftgrain->norm[i][j];
                }
                aubio_fft_rdo(fft->fft,fft->spec[i],out->data[i],fft->winsize);
        }
}

void del_aubio_mfft(aubio_mfft_t * fft) {
        uint_t i;
        for (i=0; i < fft->channels; i++)
                AUBIO_FREE(fft->spec[i]);
        AUBIO_FREE(fft->spec);
        del_aubio_fft(fft->fft);
        AUBIO_FREE(fft);        
}

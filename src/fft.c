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

static aubio_fft_t * aubio_fft_alloc(uint_t size);
static void aubio_fft_free(aubio_fft_t *s);

static aubio_fft_t * aubio_fft_alloc(uint_t size) {
	aubio_fft_t * s = AUBIO_NEW(aubio_fft_t);
	/* allocate memory */
	s->in       = AUBIO_ARRAY(real_t,size);
	s->out      = AUBIO_ARRAY(real_t,size);
	s->specdata = (fft_data_t*)fftw_malloc(sizeof(fft_data_t)*size);
	return s;
}

static void aubio_fft_free(aubio_fft_t * s) { 
	/* destroy data */
	fftw_destroy_plan(s->pfw);
	fftw_destroy_plan(s->pbw);
	if (s->specdata) 	fftw_free(s->specdata);
	if (s->out)		AUBIO_FREE(s->out);
	if (s->in )		AUBIO_FREE(s->in );
}

aubio_fft_t * new_aubio_fft(uint_t size) {
	aubio_fft_t * s =(aubio_fft_t *)aubio_fft_alloc(size);
	/* create plans */
	s->pfw = fftw_plan_dft_r2c_1d(size, s->in,  s->specdata, FFTW_ESTIMATE);
	s->pbw = fftw_plan_dft_c2r_1d(size, s->specdata, s->out, FFTW_ESTIMATE);
	return s;
}

void del_aubio_fft(aubio_fft_t * s) {
	aubio_fft_free(s);
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


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
#include "fft.h"
#include "mathutils.h"
#include "phasevoc.h"

/** phasevocoder internal object */
struct _aubio_pvoc_t {
	/** grain length */
	uint_t win_s;
	/** overlap step */
	uint_t hop_s;
	/** number of channels */
	uint_t channels;
	/** spectral data */
	aubio_fft_t * spectrum;
	/**cur output grain 		[win_s] */
	fvec_t * synth;		 
	/**last input frame 		[win_s-hop_s] */
	fvec_t * synthold; 
	/**current spectrum 		[win_s] */
	fft_data_t ** spec;
	/**current input grain		[win_s] */
	fvec_t * data;		 
	/**last input frame 		[win_s-hop_s] */
	fvec_t * dataold;  
	/** grain window		[win_s] */
	float * w;
};


/** memory allocation */
static aubio_pvoc_t * aubio_pvoc_malloc (uint_t win_s, uint_t hop_s, uint_t channels);
/** object deletion */
static void aubio_pvoc_free (aubio_pvoc_t *pv);
/** returns data and dataold slided by hop_s */
static void aubio_pvoc_swapbuffers(
		smpl_t * data,
		smpl_t * dataold,
		const smpl_t * datanew,
		uint_t win_s, uint_t hop_s);
/** do additive synthesis from 'old' and 'cur' */
static void aubio_pvoc_addsynth(
		const smpl_t * synth,
		smpl_t * synthold,
		smpl_t * synthnew, 
		uint_t win_s, uint_t hop_s);


void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t * datanew, cvec_t *fftgrain) {
	uint_t i,j;
	for (i=0; i<pv->channels; i++) {
		/* slide  */
		aubio_pvoc_swapbuffers(pv->data->data[i],pv->dataold->data[i],
				datanew->data[i],pv->win_s,pv->hop_s);
		/* windowing */
		for (j=0; j<pv->win_s; j++) pv->data->data[i][j] *= pv->w[j];
		/* fftshift */
		vec_shift(pv->data);
		/* calculate fft */
		aubio_fft_do(pv->spectrum,pv->data->data[i],pv->spec[i],pv->win_s);
		/* put norm and phase to fftgrain */
		aubio_fft_getnorm(fftgrain->norm[i], pv->spec[i], pv->win_s/2+1);
		aubio_fft_getphas(fftgrain->phas[i], pv->spec[i], pv->win_s/2+1);
	}
}

void aubio_pvoc_rdo(aubio_pvoc_t *pv,cvec_t * fftgrain, fvec_t * synthnew) {
	uint_t i,j;
	for (i=0; i<pv->channels; i++) {
		for (j=0; j<pv->win_s/2+1; j++) {
			pv->spec[i][j]  = CEXPC(I*unwrap2pi(fftgrain->phas[i][j]));
			pv->spec[i][j] *= fftgrain->norm[i][j];
		}
		aubio_fft_rdo(pv->spectrum,pv->spec[i],pv->synth->data[i],pv->win_s);
		vec_shift(pv->synth);
		for (j=0; j<pv->win_s; j++) pv->synth->data[i][j] *= pv->w[j];
		aubio_pvoc_addsynth(pv->synth->data[i],pv->synthold->data[i],
				synthnew->data[i],pv->win_s,pv->hop_s);
	}
}

void del_aubio_pvoc(aubio_pvoc_t *pv) {
	aubio_pvoc_free(pv);
}

aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s, uint_t channels) {
	aubio_pvoc_t * pv = aubio_pvoc_malloc(win_s, hop_s, channels);
	window(pv->w,pv->win_s,hanningz);
	return pv;
}

static aubio_pvoc_t * aubio_pvoc_malloc (uint_t win_s, uint_t hop_s, uint_t channels) {
	uint_t i;

	aubio_pvoc_t * pv = AUBIO_NEW(aubio_pvoc_t);

	if (win_s < 2*hop_s) {
		AUBIO_ERR("Window size is smaller than twice the hop size!\n");
		return 0;
	}

	if (hop_s < 1) {
		AUBIO_ERR("Hop size is smaller than 1!\n");
		return 0;
	}
	
	pv->spectrum = new_aubio_fft(win_s);

	/* remember old */
	pv->data     = new_fvec (win_s, channels);
	pv->synth    = new_fvec (win_s, channels);

	/* new input output */
	pv->dataold  = new_fvec  (win_s-hop_s, channels);
	pv->synthold = new_fvec (win_s-hop_s, channels);
	pv->w        = AUBIO_ARRAY(smpl_t,win_s);

	pv->spec     = AUBIO_ARRAY(fft_data_t*,channels);
	for (i=0; i<channels; i++) 
		pv->spec[i] = AUBIO_ARRAY(fft_data_t,win_s);

	pv->channels = channels;
	pv->hop_s    = hop_s;
	pv->win_s    = win_s;

	return pv;
}

static void aubio_pvoc_free (aubio_pvoc_t *pv) {
	uint_t i;
	del_aubio_fft(pv->spectrum);
	del_fvec(pv->data);
	del_fvec(pv->synth);
	del_fvec(pv->dataold);
	del_fvec(pv->synthold);
	AUBIO_FREE(pv->w);
	for (i=0; i< pv->channels; i++) {
	        AUBIO_FREE(pv->spec[i]);
	}
	AUBIO_FREE(pv->spec);
	AUBIO_FREE(pv);
}

static void aubio_pvoc_swapbuffers(smpl_t * data, smpl_t * dataold, const smpl_t * datanew, 
		uint_t win_s, uint_t hop_s)
{
	uint_t i;
	for (i=0;i<win_s-hop_s;i++)
		data[i] = dataold[i];
	for (i=0;i<hop_s;i++)
		data[win_s-hop_s+i] = datanew[i];
	for (i=0;i<win_s-hop_s;i++)
		dataold[i] = data[i+hop_s];
}

static void aubio_pvoc_addsynth(const smpl_t * synth, smpl_t * synthold, smpl_t * synthnew, 
		uint_t win_s, uint_t hop_s)
{
	uint_t i;
	smpl_t scale = 2*hop_s/(win_s+.0);
	/* add new synth to old one and put result in synthnew */
	for (i=0;i<hop_s;i++) 						
		synthnew[i] = synthold[i]+synth[i]*scale;
	/* shift synthold */
	for (i=0;i<win_s-2*hop_s;i++)	
		synthold[i] = synthold[i+hop_s];
	/* erase last frame in synthold */
	for (i=win_s-hop_s;i<win_s;i++)	
		synthold[i-hop_s]=0.;
	/* additive synth */
	for (i=0;i<win_s-hop_s;i++)	
		synthold[i] += synth[i+hop_s]*scale;
}


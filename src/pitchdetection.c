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
#include "phasevoc.h"
#include "mathutils.h"
#include "filter.h"
#include "pitchyin.h"
#include "pitchmcomb.h"
#include "pitchdetection.h"

struct _aubio_pitchdetection_t {
	aubio_pitchdetection_type type;
	aubio_pitchdetection_mode mode;
	uint_t srate;
	uint_t bufsize;
	/* for mcomb */	
	aubio_pvoc_t * pv;
	cvec_t * fftgrain; 
	aubio_pitchmcomb_t * mcomb;
	aubio_filter_t * filter;
	/* for yin */
	fvec_t * buf;
	fvec_t * yin;
};

aubio_pitchdetection_t * new_aubio_pitchdetection(uint_t bufsize, 
		uint_t hopsize, 
		uint_t channels,
		uint_t samplerate,
		aubio_pitchdetection_type type,
		aubio_pitchdetection_mode mode)
{
	aubio_pitchdetection_t *p = AUBIO_NEW(aubio_pitchdetection_t);
	p->srate = samplerate;
	p->type = type;
	p->bufsize = bufsize;
	switch(p->type) {
		case yin:
			p->buf      = new_fvec(bufsize,channels);
			p->yin      = new_fvec(bufsize/2,channels);
			break;
		case mcomb:
			p->pv       = new_aubio_pvoc(bufsize, hopsize, channels);
			p->fftgrain = new_cvec(bufsize, channels);
			p->filter   = new_aubio_adsgn_filter((smpl_t)samplerate);
			p->mcomb    = new_aubio_pitchmcomb(bufsize,channels);
			break;
		default:
			break;
	}
	return p;
}

void del_aubio_pitchdetection(aubio_pitchdetection_t * p) {
	switch(p->type) {
		case yin:
			del_fvec(p->yin);
			del_fvec(p->buf);
			break;
		case mcomb:
			del_aubio_pvoc(p->pv);
			del_cvec(p->fftgrain);
			//del_aubio_adsgn_filter(p->filter);
			//del_aubio_pitchmcomb(p->mcomb);
			break;
		default:
			break;
	}
	AUBIO_FREE(p);
}

/** \bug ugly, should replace with function pointers or so */
smpl_t aubio_pitchdetection(aubio_pitchdetection_t *p, fvec_t * ibuf) {
	smpl_t pitch = 0.;
	uint_t i,j = 0;
	switch(p->type) {
		case yin:
			/* do sliding window blocking */
			for (i=0;i<p->buf->channels;i++){
				for (j=0;j<p->buf->length-ibuf->length;j++){
					p->buf->data[i][j] = p->buf->data[i][j+ibuf->length];
				}
			}
			for (i=0;i<ibuf->channels;i++){
				for (j=0;j<ibuf->length;j++){
					p->buf->data[i][j] = ibuf->data[i][j];
				}
			}
		        pitch = aubio_pitchyin_getpitchfast(p->buf,p->yin, 0.5);
		        if (pitch>0) {
		                pitch = p->srate/(pitch+0.);
		        } else {
		                pitch = 0.;
			}
			break;
		case mcomb:
			aubio_filter_do(p->filter,ibuf);
			aubio_filter_do(p->filter,ibuf);
			aubio_pvoc_do(p->pv,ibuf,p->fftgrain);
			pitch = aubio_pitchmcomb_detect(p->mcomb,p->fftgrain);
			if (pitch>0.) {
				pitch = bintofreq(pitch,p->srate,p->bufsize);
			} else {
				pitch = 0.;
			}
			break;
		default:
			break;
	}
	return pitch;
}

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
#include "scale.h"
#include "mathutils.h" //vec_min vec_max
#include "hist.h"

/********
 * Object Structure
 */

struct _aubio_hist_t {
	/*bug: move to a fvec */
	smpl_t ** hist;
	uint_t nelems;
	uint_t channels;
	smpl_t * cent;
	aubio_scale_t *scaler;
};

/**
 * Object creation/deletion calls
 */
aubio_hist_t * new_aubio_hist (smpl_t ilow, smpl_t ihig, uint_t nelems, uint_t channels){
	aubio_hist_t * s = AUBIO_NEW(aubio_hist_t);
	smpl_t step = (ihig-ilow)/(smpl_t)(nelems);
	smpl_t accum = step;
	uint_t i;
	s->channels = channels;
	s->nelems = nelems;
	s->hist = AUBIO_ARRAY(smpl_t*, channels);
	for (i=0; i< s->channels; i++) {
		s->hist[i] = AUBIO_ARRAY(smpl_t, nelems);
	}
	s->cent = AUBIO_ARRAY(smpl_t, nelems);
	
	/* use scale to map ilow/ihig -> 0/nelems */
	s->scaler = new_aubio_scale(ilow,ihig,0,nelems);
	/* calculate centers now once */
	s->cent[0] = ilow + 0.5 * step;
	for (i=1; i < s->nelems; i++, accum+=step )
		s->cent[i] = s->cent[0] + accum;
	
	return s;	
}

void del_aubio_hist(aubio_hist_t *s) {
	uint_t i;
	for (i=0; i< s->channels; i++) {
		AUBIO_FREE(s->hist[i]);
	}
	AUBIO_FREE(s->hist);
	AUBIO_FREE(s->cent);
	del_aubio_scale(s->scaler);
	AUBIO_FREE(s);
}

/***
 * do it
 */
void aubio_hist_do (aubio_hist_t *s, fvec_t *input) 
{
	uint_t i,j;
	sint_t tmp = 0;
	aubio_scale_do(s->scaler, input);
	/* reset data */
	for (i=0; i < s->channels; i++)
		for (j=0; j < s->nelems; j++) 
			s->hist[i][j] = 0;
	/* run accum */
	for (i=0; i < input->channels; i++)
		for (j=0;  j < input->length; j++)
		{
			tmp = (sint_t)FLOOR(input->data[i][j]);
			if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
				s->hist[i][tmp] += 1;
		}
}

void aubio_hist_do_notnull (aubio_hist_t *s, fvec_t *input) 
{
	uint_t i,j;
	sint_t tmp = 0;
	aubio_scale_do(s->scaler, input);
	/* reset data */
	for (i=0; i < s->channels; i++)
		for (j=0; j < s->nelems; j++) 
			s->hist[i][j] = 0;
	/* run accum */
	for (i=0; i < input->channels; i++)
		for (j=0;  j < input->length; j++) 
		{
			if (input->data[i][j] != 0) {
				tmp = (sint_t)FLOOR(input->data[i][j]);
				if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
					s->hist[i][tmp] += 1;
			}
		}
}


void aubio_hist_dyn_notnull (aubio_hist_t *s, fvec_t *input) 
{
	uint_t i,j;
	sint_t tmp = 0;
	smpl_t ilow = vec_min(input);
	smpl_t ihig = vec_max(input);
	smpl_t step = (ihig-ilow)/(smpl_t)(s->nelems);
	
	/* readapt */
	aubio_scale_set(s->scaler, ilow, ihig, 0, s->nelems);

	/* recalculate centers */
	s->cent[0] = ilow + 0.5f * step;
	for (i=1; i < s->nelems; i++)
		s->cent[i] = s->cent[0] + i * step;

	/* scale */	
	aubio_scale_do(s->scaler, input);

	/* reset data */
	for (i=0; i < s->channels; i++)
		for (j=0; j < s->nelems; j++) 
			s->hist[i][j] = 0;
	/* run accum */
	for (i=0; i < input->channels; i++)
		for (j=0;  j < input->length; j++) 
		{
			if (input->data[i][j] != 0) {
				tmp = (sint_t)FLOOR(input->data[i][j]);
				if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
					s->hist[i][tmp] += 1;
			}
		}
}

void aubio_hist_weigth (aubio_hist_t *s) 
{
	uint_t i,j;
	for (i=0; i < s->channels; i++)
		for (j=0; j < s->nelems; j++) {
			s->hist[i][j] *= s->cent[j];
		}
}

smpl_t aubio_hist_mean (aubio_hist_t *s) 
{
	uint_t i,j;
	smpl_t tmp = 0.0f;
	for (i=0; i < s->channels; i++)
		for (j=0; j < s->nelems; j++)
			tmp += s->hist[i][j];
	return tmp/(smpl_t)(s->nelems);
}


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

struct _aubio_scale_t {
	smpl_t ilow;
	smpl_t ihig;
	smpl_t olow;
	smpl_t ohig;

	smpl_t scaler;
	smpl_t irange;
	
	/* not implemented yet : type in/out data
	bool inint;
	bool outint;
	*/
};

aubio_scale_t * new_aubio_scale (smpl_t ilow, smpl_t ihig, smpl_t olow, smpl_t ohig	){
	aubio_scale_t * s = AUBIO_NEW(aubio_scale_t);
	aubio_scale_set (s, ilow, ihig, olow, ohig);
	return s;	
}

void del_aubio_scale(aubio_scale_t *s) {
	AUBIO_FREE(s);
}

void aubio_scale_set (aubio_scale_t *s, smpl_t ilow, smpl_t ihig, smpl_t olow, smpl_t ohig) 
{
	smpl_t inputrange = ihig - ilow;
	smpl_t outputrange= ohig - olow;
	s->ilow = ilow;
	s->ihig = ihig;
	s->olow = olow;
	s->ohig = ohig;
	if (inputrange == 0 )
		s->scaler = 0.0f;
	else {
		s->scaler = outputrange/inputrange;
		if (inputrange < 0 )
			inputrange = inputrange * -1.0f;
	}
}

void aubio_scale_do (aubio_scale_t *s, fvec_t *input) 
{
	uint_t i, j;
	for (i=0; i < input->channels; i++){
		for (j=0;  j < input->length; j++){
			input->data[i][j] -= s->ilow;
			input->data[i][j] *= s->scaler;
			input->data[i][j] += s->olow;
		}
	}
}


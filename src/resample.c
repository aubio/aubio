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


#include <samplerate.h> /* from libsamplerate */

#include "aubio_priv.h"
#include "sample.h"
#include "resample.h"

struct _aubio_resampler_t {
	SRC_DATA  *proc;
	SRC_STATE *stat;
	float ratio;
	uint_t type;
};

aubio_resampler_t * new_aubio_resampler(float ratio, uint_t type) {
	aubio_resampler_t * s  = AUBIO_NEW(aubio_resampler_t);
	int error = 0;
	s->stat = src_new (type, 1, &error) ; /* only one channel */
	s->proc = AUBIO_NEW(SRC_DATA);
	if (error) AUBIO_ERR("%s\n",src_strerror(error));
	s->ratio = ratio;
	return s;
}

void del_aubio_resampler(aubio_resampler_t *s) {
	src_delete(s->stat);
	AUBIO_FREE(s->proc);
	AUBIO_FREE(s);
}

uint_t aubio_resampler_process(aubio_resampler_t *s, 
    fvec_t * input,  fvec_t * output) {
	uint_t i ;
	s->proc->input_frames = input->length;
	s->proc->output_frames = output->length;
	s->proc->src_ratio = s->ratio;
	for (i = 0 ; i< input->channels; i++) 
	{
		/* make SRC_PROC data point to input outputs */
		s->proc->data_in = input->data[i];
		s->proc->data_out= output->data[i];
		/* do resampling */
		src_process (s->stat, s->proc) ;
	}
	return AUBIO_OK;
}	

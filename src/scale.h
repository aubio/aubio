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

#ifndef SCALE_H
#define SCALE_H

#ifdef __cplusplus
extern "C" {
#endif

/* inspired from the scale object in fts (jmax) */

typedef struct _aubio_scale_t aubio_scale_t;

/* buffer function */
aubio_scale_t * new_aubio_scale(smpl_t flow, smpl_t fhig, smpl_t ilow, smpl_t ihig	);
void del_aubio_scale(aubio_scale_t *s);
void aubio_scale_do(aubio_scale_t *s,
		fvec_t * input);
void aubio_scale_set (aubio_scale_t *s, smpl_t ilow, smpl_t ihig, smpl_t olow, smpl_t ohig);

#ifdef __cplusplus
}
#endif 

#endif

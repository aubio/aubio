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
#include "fvec.h"
#include "utils/scale.h"
#include "mathutils.h" //fvec_min fvec_max
#include "utils/hist.h"

/********
 * Object Structure
 */

struct _aubio_hist_t {
  fvec_t * hist;
  uint_t nelems;
  uint_t channels;
  fvec_t * cent;
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
  s->hist = new_fvec(nelems, channels);
  s->cent = new_fvec(nelems, 1);

  /* use scale to map ilow/ihig -> 0/nelems */
  s->scaler = new_aubio_scale(ilow,ihig,0,nelems);
  /* calculate centers now once */
  s->cent->data[0][0] = ilow + 0.5 * step;
  for (i=1; i < s->nelems; i++, accum+=step )
    s->cent->data[0][i] = s->cent->data[0][0] + accum;

  return s;
}

void del_aubio_hist(aubio_hist_t *s) {
  del_fvec(s->hist);
  del_fvec(s->cent);
  del_aubio_scale(s->scaler);
  AUBIO_FREE(s);
}

/***
 * do it
 */
void aubio_hist_do (aubio_hist_t *s, fvec_t *input) {
  uint_t i,j;
  sint_t tmp = 0;
  aubio_scale_do(s->scaler, input);
  /* reset data */
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->nelems; j++)
      s->hist->data[i][j] = 0;
  /* run accum */
  for (i=0; i < input->channels; i++)
    for (j=0;  j < input->length; j++)
    {
      tmp = (sint_t)FLOOR(input->data[i][j]);
      if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
        s->hist->data[i][tmp] += 1;
    }
}

void aubio_hist_do_notnull (aubio_hist_t *s, fvec_t *input) {
  uint_t i,j;
  sint_t tmp = 0;
  aubio_scale_do(s->scaler, input);
  /* reset data */
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->nelems; j++)
      s->hist->data[i][j] = 0;
  /* run accum */
  for (i=0; i < input->channels; i++)
    for (j=0;  j < input->length; j++) {
      if (input->data[i][j] != 0) {
        tmp = (sint_t)FLOOR(input->data[i][j]);
        if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
          s->hist->data[i][tmp] += 1;
      }
    }
}


void aubio_hist_dyn_notnull (aubio_hist_t *s, fvec_t *input) {
  uint_t i,j;
  sint_t tmp = 0;
  smpl_t ilow = fvec_min(input);
  smpl_t ihig = fvec_max(input);
  smpl_t step = (ihig-ilow)/(smpl_t)(s->nelems);

  /* readapt */
  aubio_scale_set(s->scaler, ilow, ihig, 0, s->nelems);

  /* recalculate centers */
  s->cent->data[0][0] = ilow + 0.5f * step;
  for (i=1; i < s->nelems; i++)
    s->cent->data[0][i] = s->cent->data[0][0] + i * step;

  /* scale */
  aubio_scale_do(s->scaler, input);

  /* reset data */
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->nelems; j++)
      s->hist->data[i][j] = 0;
  /* run accum */
  for (i=0; i < input->channels; i++)
    for (j=0;  j < input->length; j++) {
      if (input->data[i][j] != 0) {
        tmp = (sint_t)FLOOR(input->data[i][j]);
        if ((tmp >= 0) && (tmp < (sint_t)s->nelems))
          s->hist->data[i][tmp] += 1;
      }
    }
}

void aubio_hist_weight (aubio_hist_t *s) {
  uint_t i,j;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->nelems; j++) {
      s->hist->data[i][j] *= s->cent->data[0][j];
    }
}

smpl_t aubio_hist_mean (aubio_hist_t *s) {
  uint_t i,j;
  smpl_t tmp = 0.0;
  for (i=0; i < s->channels; i++)
    for (j=0; j < s->nelems; j++)
      tmp += s->hist->data[i][j];
  return tmp/(smpl_t)(s->nelems);
}


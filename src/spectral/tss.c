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

/* default values : alpha=4, beta=3, threshold=0.25 */

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "mathutils.h"
#include "spectral/tss.h"

struct _aubio_tss_t
{
  smpl_t threshold;
  smpl_t alpha;
  smpl_t beta;
  smpl_t parm;
  smpl_t thrsfact;
  fvec_t *theta1;
  fvec_t *theta2;
  fvec_t *oft1;
  fvec_t *oft2;
  fvec_t *dev;
};

void aubio_tss_do(aubio_tss_t *o, cvec_t * input, 
    cvec_t * trans, cvec_t * stead)
{
  uint_t i,j;
  uint_t test;
  uint_t nbins     = input->length;
  uint_t channels  = input->channels;
  smpl_t alpha     = o->alpha;
  smpl_t beta      = o->beta;
  smpl_t parm      = o->parm;
  smpl_t ** dev    = (smpl_t **)o->dev->data;
  smpl_t ** oft1   = (smpl_t **)o->oft1->data;
  smpl_t ** oft2   = (smpl_t **)o->oft2->data;
  smpl_t ** theta1 = (smpl_t **)o->theta1->data;
  smpl_t ** theta2 = (smpl_t **)o->theta2->data;
  /* second phase derivative */
  for (i=0;i<channels; i++){
    for (j=0;j<nbins; j++){
      dev[i][j] = aubio_unwrap2pi(input->phas[i][j]
          -2.0*theta1[i][j]+theta2[i][j]);
      theta2[i][j] = theta1[i][j];
      theta1[i][j] = input->phas[i][j];
    }

    for (j=0;j<nbins; j++){
      /* transient analysis */
      test = (ABS(dev[i][j]) > parm*oft1[i][j]);
      trans->norm[i][j] = input->norm[i][j] * test;
      trans->phas[i][j] = input->phas[i][j] * test;
    }

    for (j=0;j<nbins; j++){
      /* steady state analysis */
      test = (ABS(dev[i][j]) < parm*oft2[i][j]);
      stead->norm[i][j] = input->norm[i][j] * test;
      stead->phas[i][j] = input->phas[i][j] * test;

      /*increase sstate probability for sines */
      test = (trans->norm[i][j]==0.);
      oft1[i][j]  = test;
      test = (stead->norm[i][j]==0.);
      oft2[i][j]  = test;
      test = (trans->norm[i][j]>0.);
      oft1[i][j] += alpha*test;
      test = (stead->norm[i][j]>0.);
      oft2[i][j] += alpha*test;
      test = (oft1[i][j]>1. && trans->norm[i][j]>0.);
      oft1[i][j] += beta*test;
      test = (oft2[i][j]>1. && stead->norm[i][j]>0.);
      oft2[i][j] += beta*test;
    }
  }
}

uint_t aubio_tss_set_threshold(aubio_tss_t *o, smpl_t threshold){
  o->threshold = threshold;
  o->parm = o->threshold * o->thrsfact;
  return AUBIO_OK;
}

aubio_tss_t * new_aubio_tss(uint_t buf_size, uint_t hop_size, uint_t channels)
{
  aubio_tss_t * o = AUBIO_NEW(aubio_tss_t);
  uint_t rsize = buf_size/2+1;
  o->threshold = 0.25;
  o->thrsfact = TWO_PI*hop_size/rsize;
  o->alpha = 3.;
  o->beta = 4.;
  o->parm = o->threshold*o->thrsfact;
  o->theta1 = new_fvec(rsize,channels);
  o->theta2 = new_fvec(rsize,channels);
  o->oft1 = new_fvec(rsize,channels);
  o->oft2 = new_fvec(rsize,channels);
  o->dev = new_fvec(rsize,channels);
  return o;
}

void del_aubio_tss(aubio_tss_t *s)
{
  free(s->theta1);
  free(s->theta2);
  free(s->oft1);
  free(s->oft2);
  free(s->dev);
  free(s);
}

uint_t aubio_tss_set_alpha(aubio_tss_t *o, smpl_t alpha){
  o->alpha = alpha;
  return AUBIO_OK;
}

uint_t aubio_tss_set_beta(aubio_tss_t *o, smpl_t beta){
  o->beta = beta;
  return AUBIO_OK;
}


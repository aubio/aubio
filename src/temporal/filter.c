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


/* Requires lsmp_t to be long or double. float will NOT give reliable 
 * results */

#include "aubio_priv.h"
#include "fvec.h"
#include "mathutils.h"
#include "temporal/filter.h"
#include "temporal/filter_priv.h"

/* bug: mono only */
void aubio_filter_do(aubio_filter_t * f, fvec_t * in) {
  uint_t i,j,l, order = f->order;
  lsmp_t *x = f->x;
  lsmp_t *y = f->y;
  lsmp_t *a = f->a;
  lsmp_t *b = f->b;
  i=0;//for (i=0;i<in->channels;i++) {
  for (j = 0; j < in->length; j++) {
    /* new input */
    //AUBIO_DBG("befor %f\t", in->data[i][j]);
    x[0] = in->data[i][j];
    y[0] = b[0] * x[0];
    for (l=1;l<order; l++) {
      y[0] += b[l] * x[l];
      y[0] -= a[l] * y[l];
    } /* + 1e-37; for denormal ? */
    /* new output */
    in->data[i][j] = y[0];
    //AUBIO_DBG("after %f\n", in->data[i][j]);
    /* store states for next sample */
    for (l=order-1; l>0; l--){
      x[l] = x[l-1];
      y[l] = y[l-1];
    }
  }
  /* store states for next buffer */
  f->x = x;
  f->y = y;
  //}	
}

void aubio_filter_do_outplace(aubio_filter_t * f, fvec_t * in, fvec_t * out) {
  uint_t i,j,l, order = f->order;
  lsmp_t *x = f->x;
  lsmp_t *y = f->y;
  lsmp_t *a = f->a;
  lsmp_t *b = f->b;

  i=0; // works in mono only !!!
  //for (i=0;i<in->channels;i++) {
  for (j = 0; j < in->length; j++) {
    /* new input */
    x[0] = in->data[i][j];
    y[0] = b[0] * x[0];
    for (l=1;l<order; l++) {
      y[0] += b[l] * x[l];
      y[0] -= a[l] * y[l];
    }
    // + 1e-37;
    /* new output */
    out->data[i][j] = y[0];
    /* store for next sample */
    for (l=order-1; l>0; l--){
      x[l] = x[l-1];
      y[l] = y[l-1];
    }
  }
  /* store for next run */
  f->x = x;
  f->y = y;
  //}
}

/*  
 *
 * despite mirroring, end effects destroy both phse and amplitude. the longer
 * the buffer, the less affected they are.
 *
 * replacing with zeros clicks.
 *
 * seems broken for order > 4 (see biquad_do_filtfilt for audible one) 
 */
void aubio_filter_do_filtfilt(aubio_filter_t * f, fvec_t * in, fvec_t * tmp) {
  uint_t j,i=0;
  uint_t length = in->length;
  //uint_t order = f->order;
  //lsmp_t mir;
  /* mirroring */
  //mir = 2*in->data[i][0];
  //for (j=1;j<order;j++)
  //f->x[j] = 0.;//mir - in->data[i][order-j];
  /* apply filtering */
  aubio_filter_do(f,in);
  /* invert */
  for (j = 0; j < length; j++)
    tmp->data[i][length-j-1] = in->data[i][j];
  /* mirror inverted */
  //mir = 2*tmp->data[i][0];
  //for (j=1;j<order;j++)
  //f->x[j] = 0.;//mir - tmp->data[i][order-j];
  /* apply filtering on inverted */
  aubio_filter_do(f,tmp);
  /* invert back */
  for (j = 0; j < length; j++)
    in->data[i][j] = tmp->data[i][length-j-1];
}

aubio_filter_t * new_aubio_filter(uint_t samplerate UNUSED, uint_t order) {
  aubio_filter_t * f = AUBIO_NEW(aubio_filter_t);
  lsmp_t * x = f->x;
  lsmp_t * y = f->y;
  lsmp_t * a = f->a;
  lsmp_t * b = f->b;
  uint_t l;
  f->order = order;
  a = AUBIO_ARRAY(lsmp_t,f->order);
  b = AUBIO_ARRAY(lsmp_t,f->order);
  x = AUBIO_ARRAY(lsmp_t,f->order);
  y = AUBIO_ARRAY(lsmp_t,f->order);
  /* initial states to zeros */
  for (l=0; l<f->order; l++){
    x[l] = 0.;
    y[l] = 0.;
  }
  f->x = x;
  f->y = y;
  f->a = a;
  f->b = b;
  return f;
}

void del_aubio_filter(aubio_filter_t * f) {
  AUBIO_FREE(f->a);
  AUBIO_FREE(f->b);
  AUBIO_FREE(f->x);
  AUBIO_FREE(f->y);
  AUBIO_FREE(f);
  return;
}

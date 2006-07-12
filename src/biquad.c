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
#include "biquad.h"

/** \note this file needs to be in double or more less precision would lead to large
 * errors in the output 
 */

struct _aubio_biquad_t {
  lsmp_t a2;
  lsmp_t a3;
  lsmp_t b1;
  lsmp_t b2;
  lsmp_t b3;
  lsmp_t o1;
  lsmp_t o2;
  lsmp_t i1;
  lsmp_t i2;
};

void aubio_biquad_do(aubio_biquad_t * b, fvec_t * in) {
  uint_t i,j;
  lsmp_t i1 = b->i1;
  lsmp_t i2 = b->i2;
  lsmp_t o1 = b->o1;
  lsmp_t o2 = b->o2;
  lsmp_t a2 = b->a2;
  lsmp_t a3 = b->a3;
  lsmp_t b1 = b->b1;
  lsmp_t b2 = b->b2;
  lsmp_t b3 = b->b3;

  i=0; // works in mono only !!!
  //for (i=0;i<in->channels;i++) {
  for (j = 0; j < in->length; j++) {
    lsmp_t i0 = in->data[i][j];
    lsmp_t o0 = b1 * i0 + b2 * i1 + b3 * i2
      - a2 * o1 - a3 * o2;// + 1e-37;
    in->data[i][j] = o0;
    i2 = i1;
    i1 = i0;
    o2 = o1;
    o1 = o0;
  }
  b->i2 = i2;
  b->i1 = i1;
  b->o2 = o2;
  b->o1 = o1;
  //}
}

void aubio_biquad_do_filtfilt(aubio_biquad_t * b, fvec_t * in, fvec_t * tmp) {
  uint_t j,i=0;
  uint_t length = in->length;
  lsmp_t mir;
  /* mirroring */
  mir = 2*in->data[i][0];
  b->i1 = mir - in->data[i][2];
  b->i2 = mir - in->data[i][1];
  /* apply filtering */
  aubio_biquad_do(b,in);
  /* invert  */
  for (j = 0; j < length; j++)
    tmp->data[i][length-j-1] = in->data[i][j];
  /* mirror again */
  mir = 2*tmp->data[i][0];
  b->i1 = mir - tmp->data[i][2];
  b->i2 = mir - tmp->data[i][1];
  /* apply filtering */
  aubio_biquad_do(b,tmp);
  /* invert back */
  for (j = 0; j < length; j++)
    in->data[i][j] = tmp->data[i][length-j-1];
}

aubio_biquad_t * new_aubio_biquad(
    lsmp_t b1, lsmp_t b2, lsmp_t b3, 
    lsmp_t a2, lsmp_t a3) {
  aubio_biquad_t * b = AUBIO_NEW(aubio_biquad_t);
  b->a2 = a2;
  b->a3 = a3;
  b->b1 = b1;
  b->b2 = b2;
  b->b3 = b3;
  b->i1 = 0.;
  b->i2 = 0.;
  b->o1 = 0.;
  b->o2 = 0.;
  return b;
}

void del_aubio_biquad(aubio_biquad_t * b) {
  AUBIO_FREE(b);
}

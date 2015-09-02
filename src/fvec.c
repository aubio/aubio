/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"
#include "fvec.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

fvec_t * new_fvec( uint_t length) {
  fvec_t * s;
  if ((sint_t)length <= 0) {
    return NULL;
  }
  s = AUBIO_NEW(fvec_t);
  s->length = length;
  s->data = AUBIO_ARRAY(smpl_t, s->length);
  return s;
}

void del_fvec(fvec_t *s) {
  AUBIO_FREE(s->data);
  AUBIO_FREE(s);
}

void fvec_set_sample(fvec_t *s, smpl_t data, uint_t position) {
  s->data[position] = data;
}

smpl_t fvec_get_sample(fvec_t *s, uint_t position) {
  return s->data[position];
}

smpl_t * fvec_get_data(fvec_t *s) {
  return s->data;
}

/* helper functions */

void fvec_print(fvec_t *s) {
  uint_t j;
  for (j=0; j< s->length; j++) {
    AUBIO_MSG(AUBIO_SMPL_FMT " ", s->data[j]);
  }
  AUBIO_MSG("\n");
}

void fvec_set_all (fvec_t *s, smpl_t val) {
#ifndef HAVE_ACCELERATE
  uint_t j;
  for (j=0; j< s->length; j++) {
    s->data[j] = val;
  }
#else
#if !HAVE_AUBIO_DOUBLE
  vDSP_vfill(&val, s->data, 1, s->length);
#else /* HAVE_AUBIO_DOUBLE */
  vDSP_vfillD(&val, s->data, 1, s->length);
#endif /* HAVE_AUBIO_DOUBLE */
#endif
}

void fvec_zeros(fvec_t *s) {
#ifndef HAVE_ACCELERATE
#if HAVE_MEMCPY_HACKS
  memset(s->data, 0, s->length * sizeof(smpl_t));
#else
  fvec_set_all (s, 0.);
#endif
#else
#if !HAVE_AUBIO_DOUBLE
  vDSP_vclr(s->data, 1, s->length);
#else /* HAVE_AUBIO_DOUBLE */
  vDSP_vclrD(s->data, 1, s->length);
#endif /* HAVE_AUBIO_DOUBLE */
#endif
}

void fvec_ones(fvec_t *s) {
  fvec_set_all (s, 1.);
}

void fvec_rev(fvec_t *s) {
  uint_t j;
  for (j=0; j< FLOOR(s->length/2); j++) {
    ELEM_SWAP(s->data[j], s->data[s->length-1-j]);
  }
}

void fvec_weight(fvec_t *s, fvec_t *weight) {
#ifndef HAVE_ACCELERATE
  uint_t j;
  uint_t length = MIN(s->length, weight->length);
  for (j=0; j< length; j++) {
    s->data[j] *= weight->data[j];
  }
#else
#if !HAVE_AUBIO_DOUBLE
  vDSP_vmul(s->data, 1, weight->data, 1, s->data, 1, s->length);
#else /* HAVE_AUBIO_DOUBLE */
  vDSP_vmulD(s->data, 1, weight->data, 1, s->data, 1, s->length);
#endif /* HAVE_AUBIO_DOUBLE */
#endif /* HAVE_ACCELERATE */
}

void fvec_copy(fvec_t *s, fvec_t *t) {
  if (s->length != t->length) {
    AUBIO_ERR("trying to copy %d elements to %d elements \n",
        s->length, t->length);
    return;
  }
#ifndef HAVE_ACCELERATE
#if HAVE_MEMCPY_HACKS
  memcpy(t->data, s->data, t->length * sizeof(smpl_t));
#else
  uint_t j;
  for (j=0; j< t->length; j++) {
    t->data[j] = s->data[j];
  }
#endif
#else
#if !HAVE_AUBIO_DOUBLE
  vDSP_mmov(s->data, t->data, 1, s->length, 1, 1);
#else /* HAVE_AUBIO_DOUBLE */
  vDSP_mmovD(s->data, t->data, 1, s->length, 1, 1);
#endif /* HAVE_AUBIO_DOUBLE */
#endif /* HAVE_ACCELERATE */
}

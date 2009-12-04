/*
  Copyright (C) 2009 Paul Brossier <piem@aubio.org>

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
#include "fmat.h"

fmat_t * new_fmat (uint_t length, uint_t height) {
  fmat_t * s = AUBIO_NEW(fmat_t);
  uint_t i,j;
  s->height = height;
  s->length = length;
  s->data = AUBIO_ARRAY(smpl_t*,s->height);
  for (i=0; i< s->height; i++) {
    s->data[i] = AUBIO_ARRAY(smpl_t, s->length);
    for (j=0; j< s->length; j++) {
      s->data[i][j]=0.;
    }
  }
  return s;
}

void del_fmat (fmat_t *s) {
  uint_t i;
  for (i=0; i<s->height; i++) {
    AUBIO_FREE(s->data[i]);
  }
  AUBIO_FREE(s->data);
  AUBIO_FREE(s);
}

void fmat_write_sample(fmat_t *s, smpl_t data, uint_t channel, uint_t position) {
  s->data[channel][position] = data;
}
smpl_t fmat_read_sample(fmat_t *s, uint_t channel, uint_t position) {
  return s->data[channel][position];
}
void fmat_put_channel(fmat_t *s, smpl_t * data, uint_t channel) {
  s->data[channel] = data;
}
smpl_t * fmat_get_channel(fmat_t *s, uint_t channel) {
  return s->data[channel];
}

smpl_t ** fmat_get_data(fmat_t *s) {
  return s->data;
}

/* helper functions */

void fmat_print(fmat_t *s) {
  uint_t i,j;
  for (i=0; i< s->height; i++) {
    for (j=0; j< s->length; j++) {
      AUBIO_MSG(AUBIO_SMPL_FMT " ", s->data[i][j]);
    }
    AUBIO_MSG("\n");
  }
}

void fmat_set(fmat_t *s, smpl_t val) {
  uint_t i,j;
  for (i=0; i< s->height; i++) {
    for (j=0; j< s->length; j++) {
      s->data[i][j] = val;
    }
  }
}

void fmat_zeros(fmat_t *s) {
  fmat_set(s, 0.);
}

void fmat_ones(fmat_t *s) {
  fmat_set(s, 1.);
}

void fmat_rev(fmat_t *s) {
  uint_t i,j;
  for (i=0; i< s->height; i++) {
    for (j=0; j< FLOOR(s->length/2); j++) {
      ELEM_SWAP(s->data[i][j], s->data[i][s->length-1-j]);
    }
  }
}

void fmat_weight(fmat_t *s, fmat_t *weight) {
  uint_t i,j;
  uint_t length = MIN(s->length, weight->length);
  for (i=0; i< s->height; i++) {
    for (j=0; j< length; j++) {
      s->data[i][j] *= weight->data[0][j];
    }
  }
}

void fmat_copy(fmat_t *s, fmat_t *t) {
  uint_t i,j;
  uint_t height = MIN(s->height, t->height);
  uint_t length = MIN(s->length, t->length);
  if (s->height != t->height) {
    AUBIO_ERR("warning, trying to copy %d rows to %d rows \n", 
            s->height, t->height);
  }
  if (s->length != t->length) {
    AUBIO_ERR("warning, trying to copy %d columns to %d columns\n", 
            s->length, t->length);
  }
  for (i=0; i< height; i++) {
    for (j=0; j< length; j++) {
      t->data[i][j] = s->data[i][j];
    }
  }
}


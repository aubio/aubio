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

fvec_t * new_fvec( uint_t length, uint_t channels) {
  fvec_t * s = AUBIO_NEW(fvec_t);
  uint_t i,j;
  s->channels = channels;
  s->length = length;
  s->data = AUBIO_ARRAY(smpl_t*,s->channels);
  for (i=0; i< s->channels; i++) {
    s->data[i] = AUBIO_ARRAY(smpl_t, s->length);
    for (j=0; j< s->length; j++) {
      s->data[i][j]=0.;
    }
  }
  return s;
}

void del_fvec(fvec_t *s) {
  uint_t i;
  for (i=0; i<s->channels; i++) {
    AUBIO_FREE(s->data[i]);
  }
  AUBIO_FREE(s->data);
  AUBIO_FREE(s);
}

void fvec_write_sample(fvec_t *s, smpl_t data, uint_t channel, uint_t position) {
  s->data[channel][position] = data;
}
smpl_t fvec_read_sample(fvec_t *s, uint_t channel, uint_t position) {
  return s->data[channel][position];
}
void fvec_put_channel(fvec_t *s, smpl_t * data, uint_t channel) {
  s->data[channel] = data;
}
smpl_t * fvec_get_channel(fvec_t *s, uint_t channel) {
  return s->data[channel];
}

smpl_t ** fvec_get_data(fvec_t *s) {
  return s->data;
}

/* helper functions */

void fvec_print(fvec_t *s) {
  uint_t i,j;
  for (i=0; i< s->channels; i++) {
    for (j=0; j< s->length; j++) {
      AUBIO_MSG(AUBIO_SMPL_FMT " ", s->data[i][j]);
    }
    AUBIO_MSG("\n");
  }
}

void fvec_set(fvec_t *s, smpl_t val) {
  uint_t i,j;
  for (i=0; i< s->channels; i++) {
    for (j=0; j< s->length; j++) {
      s->data[i][j] = val;
    }
  }
}

void fvec_zeros(fvec_t *s) {
  fvec_set(s, 0.);
}

void fvec_ones(fvec_t *s) {
  fvec_set(s, 1.);
}

void fvec_rev(fvec_t *s) {
  uint_t i,j;
  for (i=0; i< s->channels; i++) {
    for (j=0; j< FLOOR(s->length/2); j++) {
      ELEM_SWAP(s->data[i][j], s->data[i][s->length-1-j]);
    }
  }
}

void fvec_weight(fvec_t *s, fvec_t *weight) {
  uint_t i,j;
  uint_t length = MIN(s->length, weight->length);
  for (i=0; i< s->channels; i++) {
    for (j=0; j< length; j++) {
      s->data[i][j] *= weight->data[0][j];
    }
  }
}

void fvec_copy(fvec_t *s, fvec_t *t) {
  uint_t i,j;
  uint_t channels = MIN(s->channels, t->channels);
  uint_t length = MIN(s->length, t->length);
  if (s->channels != t->channels) {
    AUBIO_ERR("warning, trying to copy %d channels to %d channels\n", 
            s->channels, t->channels);
  }
  if (s->length != t->length) {
    AUBIO_ERR("warning, trying to copy %d elements to %d elements \n", 
            s->length, t->length);
  }
  for (i=0; i< channels; i++) {
    for (j=0; j< length; j++) {
      t->data[i][j] = s->data[i][j];
    }
  }
}


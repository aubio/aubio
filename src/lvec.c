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
#include "lvec.h"

lvec_t * new_lvec( uint_t length) {
  lvec_t * s = AUBIO_NEW(lvec_t);
  uint_t j;
  s->length = length;
  s->data = AUBIO_ARRAY(lsmp_t, s->length);
  for (j=0; j< s->length; j++) {
    s->data[j]=0.;
  }
  return s;
}

void del_lvec(lvec_t *s) {
  AUBIO_FREE(s->data);
  AUBIO_FREE(s);
}

void lvec_write_sample(lvec_t *s, lsmp_t data, uint_t position) {
  s->data[position] = data;
}
lsmp_t lvec_read_sample(lvec_t *s, uint_t position) {
  return s->data[position];
}

lsmp_t * lvec_get_data(lvec_t *s) {
  return s->data;
}

/* helper functions */

void lvec_print(lvec_t *s) {
  uint_t j;
  for (j=0; j< s->length; j++) {
    AUBIO_MSG(AUBIO_LSMP_FMT " ", s->data[j]);
  }
  AUBIO_MSG("\n");
}

void lvec_set(lvec_t *s, smpl_t val) {
  uint_t j;
  for (j=0; j< s->length; j++) {
    s->data[j] = val;
  }
}

void lvec_zeros(lvec_t *s) {
  lvec_set(s, 0.);
}

void lvec_ones(lvec_t *s) {
  lvec_set(s, 1.);
}


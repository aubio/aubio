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
#include "cvec.h"

cvec_t * new_cvec( uint_t length) {
  cvec_t * s = AUBIO_NEW(cvec_t);
  s->length = length/2 + 1;
  s->norm = AUBIO_ARRAY(smpl_t,s->length);
  s->phas = AUBIO_ARRAY(smpl_t,s->length);
  return s;
}

void del_cvec(cvec_t *s) {
  AUBIO_FREE(s->norm);
  AUBIO_FREE(s->phas);
  AUBIO_FREE(s);
}

void cvec_write_norm(cvec_t *s, smpl_t data, uint_t position) {
  s->norm[position] = data;
}
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t position) {
  s->phas[position] = data;
}
smpl_t cvec_read_norm(cvec_t *s, uint_t position) {
  return s->norm[position];
}
smpl_t cvec_read_phas(cvec_t *s, uint_t position) {
  return s->phas[position];
}
smpl_t * cvec_get_norm(cvec_t *s) {
  return s->norm;
}
smpl_t * cvec_get_phas(cvec_t *s) {
  return s->phas;
}

/* helper functions */

void cvec_print(cvec_t *s) {
  uint_t j;
  AUBIO_MSG("norm: ");
  for (j=0; j< s->length; j++) {
    AUBIO_MSG(AUBIO_SMPL_FMT " ", s->norm[j]);
  }
  AUBIO_MSG("\n");
  AUBIO_MSG("phas: ");
  for (j=0; j< s->length; j++) {
    AUBIO_MSG(AUBIO_SMPL_FMT " ", s->phas[j]);
  }
  AUBIO_MSG("\n");
}

void cvec_set(cvec_t *s, smpl_t val) {
  uint_t j;
  for (j=0; j< s->length; j++) {
    s->norm[j] = val;
  }
}

void cvec_zeros(cvec_t *s) {
  cvec_set(s, 0.);
}

void cvec_ones(cvec_t *s) {
  cvec_set(s, 1.);
}


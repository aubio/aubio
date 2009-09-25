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

lvec_t * new_lvec( uint_t length, uint_t channels) {
  lvec_t * s = AUBIO_NEW(lvec_t);
  uint_t i,j;
  s->channels = channels;
  s->length = length;
  s->data = AUBIO_ARRAY(lsmp_t*,s->channels);
  for (i=0; i< s->channels; i++) {
    s->data[i] = AUBIO_ARRAY(lsmp_t, s->length);
    for (j=0; j< s->length; j++) {
      s->data[i][j]=0.;
    }
  }
  return s;
}

void del_lvec(lvec_t *s) {
  uint_t i;
  for (i=0; i<s->channels; i++) {
    AUBIO_FREE(s->data[i]);
  }
  AUBIO_FREE(s->data);
  AUBIO_FREE(s);
}

void lvec_write_sample(lvec_t *s, lsmp_t data, uint_t channel, uint_t position) {
  s->data[channel][position] = data;
}
lsmp_t lvec_read_sample(lvec_t *s, uint_t channel, uint_t position) {
  return s->data[channel][position];
}
void lvec_put_channel(lvec_t *s, lsmp_t * data, uint_t channel) {
  s->data[channel] = data;
}
lsmp_t * lvec_get_channel(lvec_t *s, uint_t channel) {
  return s->data[channel];
}

lsmp_t ** lvec_get_data(lvec_t *s) {
  return s->data;
}

/* helper functions */

void lvec_print(lvec_t *s) {
  uint_t i,j;
  for (i=0; i< s->channels; i++) {
    for (j=0; j< s->length; j++) {
      AUBIO_MSG(AUBIO_LSMP_FMT " ", s->data[i][j]);
    }
    AUBIO_MSG("\n");
  }
}


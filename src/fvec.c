/*
   Copyright (C) 2003-2007 Paul Brossier <piem@piem.org>

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

void fvec_print(fvec_t *s) {
  uint_t i,j;
  for (i=0; i< s->channels; i++) {
    for (j=0; j< s->length; j++) {
      AUBIO_MSG("%f ", s->data[i][j]);
    }
    AUBIO_MSG("\n");
  }
}

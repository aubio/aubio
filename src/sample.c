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

cvec_t * new_cvec( uint_t length, uint_t channels) {
  cvec_t * s = AUBIO_NEW(cvec_t);
  uint_t i,j;
  s->channels = channels;
  s->length = length/2 + 1;
  s->norm = AUBIO_ARRAY(smpl_t*,s->channels);
  s->phas = AUBIO_ARRAY(smpl_t*,s->channels);
  for (i=0; i< s->channels; i++) {
    s->norm[i] = AUBIO_ARRAY(smpl_t,s->length);
    s->phas[i] = AUBIO_ARRAY(smpl_t,s->length);
    for (j=0; j< s->length; j++) {
      s->norm[i][j]=0.;
      s->phas[i][j]=0.;
    }
  }
  return s;
}

void del_cvec(cvec_t *s) {
  uint_t i;
  for (i=0; i<s->channels; i++) {
    AUBIO_FREE(s->norm[i]);
    AUBIO_FREE(s->phas[i]);
  }
  AUBIO_FREE(s->norm);
  AUBIO_FREE(s->phas);
  AUBIO_FREE(s);
}

void cvec_write_norm(cvec_t *s, smpl_t data, uint_t channel, uint_t position) {
  s->norm[channel][position] = data;
}
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t channel, uint_t position) {
  s->phas[channel][position] = data;
}
smpl_t cvec_read_norm(cvec_t *s, uint_t channel, uint_t position) {
  return s->norm[channel][position];
}
smpl_t cvec_read_phas(cvec_t *s, uint_t channel, uint_t position) {
  return s->phas[channel][position];
}
void cvec_put_norm_channel(cvec_t *s, smpl_t * data, uint_t channel) {
  s->norm[channel] = data;
}
void cvec_put_phas_channel(cvec_t *s, smpl_t * data, uint_t channel) {
  s->phas[channel] = data;
}
smpl_t * cvec_get_norm_channel(cvec_t *s, uint_t channel) {
  return s->norm[channel];
}
smpl_t * cvec_get_phas_channel(cvec_t *s, uint_t channel) {
  return s->phas[channel];
}
smpl_t ** cvec_get_norm(cvec_t *s) {
  return s->norm;
}
smpl_t ** cvec_get_phas(cvec_t *s) {
  return s->phas;
}

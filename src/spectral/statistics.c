/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>

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
#include "spectral/specdesc.h"

smpl_t
cvec_sum_channel (cvec_t * s, uint_t i)
{
  uint_t j;
  smpl_t tmp = 0.0;
  for (j = 0; j < s->length; j++)
      tmp += s->norm[i][j];
  return tmp;
}

smpl_t
cvec_mean_channel (cvec_t * s, uint_t i)
{
  return cvec_sum_channel(s, i) / (smpl_t) (s->length);
}

smpl_t
cvec_centroid_channel (cvec_t * spec, uint_t i)
{
  smpl_t sum = 0., sc = 0.;
  uint_t j;
  sum = cvec_sum_channel (spec, i); 
  if (sum == 0.) {
    return 0.;
  } else {
    for (j = 0; j < spec->length; j++) {
      sc += (smpl_t) j *spec->norm[i][j];
    }
    return sc / sum;
  }
}

smpl_t
cvec_moment_channel (cvec_t * spec, uint_t i, uint_t order)
{
  smpl_t sum = 0., centroid = 0., sc = 0.;
  uint_t j;
  sum = cvec_sum_channel (spec, i); 
  if (sum == 0.) {
    return 0.;
  } else {
    centroid = cvec_centroid_channel (spec, i);
    for (j = 0; j < spec->length; j++) {
      sc += (smpl_t) POW(j - centroid, order) * spec->norm[i][j];
    }
    return sc / sum;
  }
}

void
aubio_specdesc_centroid (aubio_specdesc_t * o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i;
  for (i = 0; i < spec->channels; i++) {
    desc->data[i][0] = cvec_centroid_channel (spec, i); 
  }
}

void
aubio_specdesc_spread (aubio_specdesc_t * o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i;
  for (i = 0; i < spec->channels; i++) {
    desc->data[i][0] = cvec_moment_channel (spec, i, 2);
  }
}

void
aubio_specdesc_skewness (aubio_specdesc_t * o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i; smpl_t spread;
  for (i = 0; i < spec->channels; i++) {
    spread = cvec_moment_channel (spec, i, 2);
    if (spread == 0) {
      desc->data[i][0] = 0.;
    } else {
      desc->data[i][0] = cvec_moment_channel (spec, i, 3);
      desc->data[i][0] /= POW ( SQRT (spread), 3);
    }
  }
}

void
aubio_specdesc_kurtosis (aubio_specdesc_t * o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i; smpl_t spread;
  for (i = 0; i < spec->channels; i++) {
    spread = cvec_moment_channel (spec, i, 2);
    if (spread == 0) {
      desc->data[i][0] = 0.;
    } else {
      desc->data[i][0] = cvec_moment_channel (spec, i, 4);
      desc->data[i][0] /= SQR (spread);
    }
  }
}

void
aubio_specdesc_slope (aubio_specdesc_t * o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i, j;
  smpl_t norm = 0, sum = 0.; 
  // compute N * sum(j**2) - sum(j)**2
  for (j = 0; j < spec->length; j++) {
    norm += j*j;
  }
  norm *= spec->length;
  // sum_0^N(j) = length * (length + 1) / 2
  norm -= SQR( (spec->length) * (spec->length - 1.) / 2. );
  for (i = 0; i < spec->channels; i++) {
    sum = cvec_sum_channel (spec, i); 
    desc->data[i][0] = 0.;
    if (sum == 0.) {
      break; 
    } else {
      for (j = 0; j < spec->length; j++) {
        desc->data[i][0] += j * spec->norm[i][j]; 
      }
      desc->data[i][0] *= spec->length;
      desc->data[i][0] -= sum * spec->length * (spec->length - 1) / 2.;
      desc->data[i][0] /= norm;
      desc->data[i][0] /= sum;
    }
  }
}

void
aubio_specdesc_decrease (aubio_specdesc_t *o UNUSED, cvec_t * spec,
    fvec_t * desc)
{
  uint_t i, j; smpl_t sum;
  for (i = 0; i < spec->channels; i++) {
    sum = cvec_sum_channel (spec, i); 
    desc->data[i][0] = 0;
    if (sum == 0.) {
      break;
    } else {
      sum -= spec->norm[i][0];
      for (j = 1; j < spec->length; j++) {
        desc->data[i][0] += (spec->norm[i][j] - spec->norm[i][0]) / j;
      }
      desc->data[i][0] /= sum;
    }
  }
}

void
aubio_specdesc_rolloff (aubio_specdesc_t *o UNUSED, cvec_t * spec,
    fvec_t *desc)
{
  uint_t i, j; smpl_t cumsum, rollsum;
  for (i = 0; i < spec->channels; i++) {
    cumsum = 0.; rollsum = 0.;
    for (j = 0; j < spec->length; j++) {
      cumsum += SQR (spec->norm[i][j]);
    }
    if (cumsum == 0) {
      desc->data[i][0] = 0.;
    } else {
      cumsum *= 0.95;
      j = 0;
      while (rollsum < cumsum) { 
        rollsum += SQR (spec->norm[i][j]);
        j++;
      }
      desc->data[i][0] = j;
    }
  }
}

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

void aubio_specdesc_centroid (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_spread (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_skewness (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_kurtosis (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_slope (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_decrease (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);
void aubio_specdesc_rolloff (aubio_specdesc_t * o, const cvec_t * spec,
    fvec_t * desc);


smpl_t cvec_sum (const cvec_t * s);
smpl_t cvec_mean (const cvec_t * s);
smpl_t cvec_centroid (const cvec_t * s);
smpl_t cvec_moment (const cvec_t * s, uint_t moment);

smpl_t
cvec_sum (const cvec_t * s)
{
  uint_t j;
  smpl_t tmp = 0.0;
  for (j = 0; j < s->length; j++) {
    tmp += s->norm[j];
  }
  return tmp;
}

smpl_t
cvec_mean (const cvec_t * s)
{
  return cvec_sum (s) / (smpl_t) (s->length);
}

smpl_t
cvec_centroid (const cvec_t * spec)
{
  smpl_t sum = 0., sc = 0.;
  uint_t j;
  sum = cvec_sum (spec);
  if (sum == 0.) {
    return 0.;
  } else {
    for (j = 0; j < spec->length; j++) {
      sc += (smpl_t) j *spec->norm[j];
    }
    return sc / sum;
  }
}

smpl_t
cvec_moment (const cvec_t * spec, uint_t order)
{
  smpl_t sum = 0., centroid = 0., sc = 0.;
  uint_t j;
  sum = cvec_sum (spec);
  if (sum == 0.) {
    return 0.;
  } else {
    centroid = cvec_centroid (spec);
    for (j = 0; j < spec->length; j++) {
      sc += POW((smpl_t)j - centroid, (smpl_t)order) * spec->norm[j];
    }
    return sc / sum;
  }
}

void
aubio_specdesc_centroid (aubio_specdesc_t * o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  desc->data[0] = cvec_centroid (spec);
}

void
aubio_specdesc_spread (aubio_specdesc_t * o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  desc->data[0] = cvec_moment (spec, 2);
}

void
aubio_specdesc_skewness (aubio_specdesc_t * o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  smpl_t spread;
  spread = cvec_moment (spec, 2);
  if (spread == 0) {
    desc->data[0] = 0.;
  } else {
    desc->data[0] = cvec_moment (spec, 3);
    desc->data[0] /= POW ( SQRT (spread), 3);
  }
}

void
aubio_specdesc_kurtosis (aubio_specdesc_t * o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  smpl_t spread;
  spread = cvec_moment (spec, 2);
  if (spread == 0) {
    desc->data[0] = 0.;
  } else {
    desc->data[0] = cvec_moment (spec, 4);
    desc->data[0] /= SQR (spread);
  }
}

void
aubio_specdesc_slope (aubio_specdesc_t * o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  uint_t j;
  const double n = spec->length;
  // N * sum(j**2) - sum(j)**2, for j in [0, N-1], in closed form:
  // sum(j**2) = (N-1) N (2N-1) / 6 and sum(j) = N (N-1) / 2
  const smpl_t norm = (smpl_t)(n * (n - 1.) * n * (2. * n - 1.) / 6.
      - SQR(n * (n - 1.) / 2.));
  smpl_t sum = 0;
  sum = cvec_sum (spec);
  desc->data[0] = 0.;
  if (sum == 0.) {
    return;
  } else {
    for (j = 0; j < spec->length; j++) {
      desc->data[0] += (smpl_t)j * spec->norm[j];
    }
    desc->data[0] *= (smpl_t)spec->length;
    desc->data[0] -= sum * (smpl_t)(n * (n - 1.) / 2.);
    desc->data[0] /= norm;
    desc->data[0] /= sum;
  }
}

void
aubio_specdesc_decrease (aubio_specdesc_t *o UNUSED, const cvec_t * spec,
    fvec_t * desc)
{
  uint_t j; smpl_t sum;
  sum = cvec_sum (spec);
  desc->data[0] = 0;
  if (sum == 0.) {
    return;
  } else {
    sum -= spec->norm[0];
    for (j = 1; j < spec->length; j++) {
      desc->data[0] += (spec->norm[j] - spec->norm[0]) / (smpl_t)j;
    }
    desc->data[0] /= sum;
  }
}

void
aubio_specdesc_rolloff (aubio_specdesc_t *o UNUSED, const cvec_t * spec,
    fvec_t *desc)
{
  uint_t j; smpl_t cumsum, rollsum;
  cumsum = 0.; rollsum = 0.;
  for (j = 0; j < spec->length; j++) {
    cumsum += SQR (spec->norm[j]);
  }
  if (cumsum == 0) {
    desc->data[0] = 0.;
  } else {
    cumsum *= (smpl_t)0.95;
    // j counts the bins summed so far; the result is the index of the bin
    // that reaches 95% of the energy, in [0, length - 1] (upstream gh-318).
    // The bound keeps rounding from walking off the end of norm.
    j = 0;
    while (rollsum < cumsum && j < spec->length) {
      rollsum += SQR (spec->norm[j]);
      j++;
    }
    desc->data[0] = (smpl_t)(MAX (1, j) - 1);
  }
}

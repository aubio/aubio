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


/* Requires lsmp_t to be long or double. float will NOT give reliable
 * results */

#include "aubio_priv.h"
#include "fvec.h"
#include "lvec.h"
#include "mathutils.h"
#include "temporal/filter.h"

struct _aubio_filter_t
{
  uint_t order;
  uint_t samplerate;
  lvec_t *a;
  lvec_t *b;
  lvec_t *y;
  lvec_t *x;
};

void
aubio_filter_do_outplace (aubio_filter_t * f, const fvec_t * in, fvec_t * out)
{
  fvec_copy (in, out);
  aubio_filter_do (f, out);
}

void
aubio_filter_do (aubio_filter_t * f, fvec_t * in)
{
  uint_t j, l, order = f->order;
  lsmp_t *x = f->x->data;
  lsmp_t *y = f->y->data;
  lsmp_t *a = f->a->data;
  lsmp_t *b = f->b->data;

  for (j = 0; j < in->length; j++) {
    /* new input */
    x[0] = KILL_DENORMAL (in->data[j]);
    y[0] = b[0] * x[0];
    for (l = 1; l < order; l++) {
      y[0] += b[l] * x[l];
      y[0] -= a[l] * y[l];
    }
    /* new output */
    in->data[j] = (smpl_t)y[0];
    /* store for next sample */
    for (l = order - 1; l > 0; l--) {
      x[l] = x[l - 1];
      y[l] = y[l - 1];
    }
  }
}

/* The rough way: reset memory of filter between each run to avoid end effects. */
void
aubio_filter_do_filtfilt (aubio_filter_t * f, fvec_t * in, fvec_t * tmp)
{
  uint_t j;
  uint_t length = in->length;
  /* apply filtering */
  aubio_filter_do (f, in);
  aubio_filter_do_reset (f);
  /* mirror */
  for (j = 0; j < length; j++)
    tmp->data[length - j - 1] = in->data[j];
  /* apply filtering on mirrored */
  aubio_filter_do (f, tmp);
  aubio_filter_do_reset (f);
  /* invert back */
  for (j = 0; j < length; j++)
    in->data[j] = tmp->data[length - j - 1];
}

lvec_t *
aubio_filter_get_feedback (const aubio_filter_t * f)
{
  return f->a;
}

lvec_t *
aubio_filter_get_feedforward (const aubio_filter_t * f)
{
  return f->b;
}

uint_t
aubio_filter_get_order (const aubio_filter_t * f)
{
  return f->order;
}

uint_t
aubio_filter_get_samplerate (const aubio_filter_t * f)
{
  return f->samplerate;
}

uint_t
aubio_filter_set_samplerate (aubio_filter_t * f, uint_t samplerate)
{
  f->samplerate = samplerate;
  return AUBIO_OK;
}

uint_t
aubio_filter_set_analog (aubio_filter_t * f, const lsmp_t * num,
    const lsmp_t * den, uint_t degree)
{
  /* the analog H(s) = num(s) / den(s), coefficients by ascending power of s,
     mapped to z with the bilinear transform s = 2 fs (1 - z^-1) / (1 + z^-1):
     each s^i becomes (2 fs)^i (1 - z^-1)^i (1 + z^-1)^(degree - i), once
     both sides are multiplied by (1 + z^-1)^degree */
  lsmp_t term[AUBIO_FILTER_MAX_ANALOG_DEGREE + 1];
  lsmp_t *b = f->b->data, *a = f->a->data;
  lsmp_t two_fs = 2. * f->samplerate, scale = 1.;
  uint_t i, j, k;
  if (degree > AUBIO_FILTER_MAX_ANALOG_DEGREE || f->order != degree + 1
      || f->samplerate == 0) {
    AUBIO_ERR("filter: can not set a degree %d analog design on an order %d "
        "filter at %dHz\n", degree, f->order, f->samplerate);
    return AUBIO_FAIL;
  }
  for (j = 0; j <= degree; j++) {
    b[j] = 0.;
    a[j] = 0.;
  }
  for (i = 0; i <= degree; i++) {
    /* term = (1 - x)^i (1 + x)^(degree - i), x = z^-1 */
    term[0] = 1.;
    for (j = 1; j <= degree; j++) term[j] = 0.;
    for (k = 0; k < degree; k++) {
      lsmp_t sign = (k < i) ? -1. : 1.;
      for (j = k + 1; j > 0; j--) term[j] += sign * term[j - 1];
    }
    for (j = 0; j <= degree; j++) {
      b[j] += num[i] * scale * term[j];
      a[j] += den[i] * scale * term[j];
    }
    scale *= two_fs;
  }
  if (a[0] == 0.) {
    AUBIO_ERR("filter: analog design has no denominator\n");
    return AUBIO_FAIL;
  }
  for (j = degree + 1; j > 0; j--) {
    b[j - 1] /= a[0];
    a[j - 1] /= a[0];
  }
  return AUBIO_OK;
}

void
aubio_filter_do_reset (aubio_filter_t * f)
{
  lvec_zeros (f->x);
  lvec_zeros (f->y);
}

aubio_filter_t *
new_aubio_filter (uint_t order)
{
  aubio_filter_t *f = AUBIO_NEW (aubio_filter_t);

  /* validate order parameter to prevent unrealistic allocations */
  if ((sint_t)order < 1) {
    AUBIO_FREE(f);
    return NULL;
  }
  /* typical values are 3, 5, or 7; allow up to 512 as reasonable upper bound */
  if (order > 512) {
    AUBIO_ERR("filter: order %d is unrealistic (max 512), aborting\n", order);
    AUBIO_FREE(f);
    return NULL;
  }

  /* check if main structure allocation succeeded */
  if (!f) {
    return NULL;
  }
  /* allocate filter buffers and check each allocation */
  f->x = new_lvec (order);
  if (!f->x) goto beach;
  f->y = new_lvec (order);
  if (!f->y) goto beach;
  f->a = new_lvec (order);
  if (!f->a) goto beach;
  f->b = new_lvec (order);
  if (!f->b) goto beach;
  /* by default, samplerate is not set */
  f->samplerate = 0;
  f->order = order;
  /* set default to identity */
  f->a->data[0] = 1.;
  f->b->data[0] = 1.;
  return f;

beach:
  /* cleanup on allocation failure */
  if (f->a) del_lvec(f->a);
  if (f->b) del_lvec(f->b);
  if (f->x) del_lvec(f->x);
  if (f->y) del_lvec(f->y);
  AUBIO_FREE(f);
  return NULL;
}

void
del_aubio_filter (aubio_filter_t * f)
{
  del_lvec (f->a);
  del_lvec (f->b);
  del_lvec (f->x);
  del_lvec (f->y);
  AUBIO_FREE (f);
  return;
}

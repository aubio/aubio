/*
  Copyright (C) 2018 Paul Brossier <piem@aubio.org>

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
#include "tensor.h"
#include "batchnorm.h"

struct _aubio_batchnorm_t {
  uint_t n_outputs;
  fvec_t *gamma;
  fvec_t *beta;
  fvec_t *moving_mean;
  fvec_t *moving_variance;
};

#if defined(DEBUG)
static void aubio_batchnorm_debug(aubio_batchnorm_t *c,
    aubio_tensor_t *input_tensor);
#endif

aubio_batchnorm_t *new_aubio_batchnorm(void)
{
  aubio_batchnorm_t *c = AUBIO_NEW(aubio_batchnorm_t);
  // note; no input parameter, so no other possible failure
  return c;
}

static void aubio_batchnorm_reset(aubio_batchnorm_t *c) {
  AUBIO_ASSERT(c);
  if (c->gamma)
    del_fvec(c->gamma);
  if (c->beta)
    del_fvec(c->beta);
  if (c->moving_mean)
    del_fvec(c->moving_mean);
  if (c->moving_variance)
    del_fvec(c->moving_variance);
}

void del_aubio_batchnorm(aubio_batchnorm_t* c) {
  aubio_batchnorm_reset(c);
  AUBIO_FREE(c);
}

#if defined(DEBUG)
void aubio_batchnorm_debug(aubio_batchnorm_t *c, aubio_tensor_t *input_tensor)
{
  AUBIO_DBG("batchnorm: %15s -> %s (%d params) (4 * (%d,))\n",
      aubio_tensor_get_shape_string(input_tensor),
      aubio_tensor_get_shape_string(input_tensor), // same output shape
      c->n_outputs, 4 * c->n_outputs);
}
#endif

uint_t aubio_batchnorm_get_output_shape(aubio_batchnorm_t *c,
    aubio_tensor_t *input, uint_t *shape)
{
  uint_t i;

  AUBIO_ASSERT(c && input && shape);

  for (i = 0; i < input->ndim; i++) {
    shape[i] = input->shape[i];
  }

  aubio_batchnorm_reset(c);

  c->n_outputs = input->shape[input->ndim - 1];

  c->gamma = new_fvec(c->n_outputs);
  c->beta = new_fvec(c->n_outputs);
  c->moving_mean = new_fvec(c->n_outputs);
  c->moving_variance = new_fvec(c->n_outputs);

  if (!c->gamma || !c->beta || !c->moving_mean || !c->moving_variance)
  {
    aubio_batchnorm_reset(c);
    return AUBIO_FAIL;
  }

#if defined(DEBUG)
  aubio_batchnorm_debug(c, input);
#endif

  return AUBIO_OK;
}

void aubio_batchnorm_do(aubio_batchnorm_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  smpl_t s;
  uint_t i, j;
  uint_t ii = 0;
  uint_t length = activations->shape[activations->ndim - 1];
  uint_t height = activations->size / length;

  AUBIO_ASSERT(c);
  AUBIO_ASSERT_EQUAL_SHAPE(input_tensor, activations);
  AUBIO_ASSERT(length == c->n_outputs);
  AUBIO_ASSERT(height * length == activations->size);

  for (i = 0; i < height; i++) {
    for (j = 0; j < length; j++) {
      s = input_tensor->buffer[ii + j];
      s -= c->moving_mean->data[j];
      s *= c->gamma->data[j];
      s /= SQRT(c->moving_variance->data[j] + 1.e-4);
      s += c->beta->data[j];
      activations->buffer[ii + j] = s;
    }
    ii += length;
  }
}

uint_t aubio_batchnorm_set_gamma(aubio_batchnorm_t *t, fvec_t *gamma)
{
  AUBIO_ASSERT(t && t->gamma);
  AUBIO_ASSERT(gamma);
  if (t->gamma->length != gamma->length) return AUBIO_FAIL;
  fvec_copy(gamma, t->gamma);
  return AUBIO_OK;
}

uint_t aubio_batchnorm_set_beta(aubio_batchnorm_t *t, fvec_t *beta)
{
  AUBIO_ASSERT(t && t->beta && beta);
  if (t->beta->length != beta->length)
    return AUBIO_FAIL;
  fvec_copy(beta, t->beta);
  return AUBIO_OK;
}

uint_t aubio_batchnorm_set_moving_mean(aubio_batchnorm_t *t,
    fvec_t *moving_mean)
{
  AUBIO_ASSERT(t && t->moving_mean);
  AUBIO_ASSERT(moving_mean);
  if (t->moving_mean->length != moving_mean->length)
    return AUBIO_FAIL;
  fvec_copy(moving_mean, t->moving_mean);
  return AUBIO_OK;
}

uint_t aubio_batchnorm_set_moving_variance(aubio_batchnorm_t *t,
    fvec_t *moving_variance)
{
  AUBIO_ASSERT(t && t->moving_variance);
  AUBIO_ASSERT(moving_variance);
  if (t->moving_variance->length != moving_variance->length)
    return AUBIO_FAIL;
  fvec_copy(moving_variance, t->moving_variance);
  return AUBIO_OK;
}

fvec_t *aubio_batchnorm_get_gamma(aubio_batchnorm_t *t)
{
  AUBIO_ASSERT(t && t->gamma);
  return t->gamma;
}

fvec_t *aubio_batchnorm_get_beta(aubio_batchnorm_t *t)
{
  AUBIO_ASSERT(t && t->beta);
  return t->beta;
}

fvec_t *aubio_batchnorm_get_moving_mean(aubio_batchnorm_t *t)
{
  AUBIO_ASSERT(t && t->moving_mean);
  return t->moving_mean;
}

fvec_t *aubio_batchnorm_get_moving_variance(aubio_batchnorm_t *t)
{
  AUBIO_ASSERT(t && t->moving_variance);
  return t->moving_variance;
}

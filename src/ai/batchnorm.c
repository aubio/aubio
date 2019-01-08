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

static void aubio_batchnorm_debug(aubio_batchnorm_t *c,
    aubio_tensor_t *input_tensor);

aubio_batchnorm_t *new_aubio_batchnorm(uint_t n_outputs)
{
  aubio_batchnorm_t *c = AUBIO_NEW(aubio_batchnorm_t);

  AUBIO_GOTO_FAILURE((sint_t)n_outputs > 0);

  c->n_outputs = n_outputs;

  c->gamma = new_fvec(n_outputs);
  c->beta = new_fvec(n_outputs);
  c->moving_mean = new_fvec(n_outputs);
  c->moving_variance = new_fvec(n_outputs);

  return c;

failure:
  del_aubio_batchnorm(c);
  return NULL;
}

void del_aubio_batchnorm(aubio_batchnorm_t* c) {
  AUBIO_ASSERT(c);
  if (c->gamma)
    del_fvec(c->gamma);
  if (c->beta)
    del_fvec(c->beta);
  if (c->moving_mean)
    del_fvec(c->moving_mean);
  if (c->moving_variance)
    del_fvec(c->moving_variance);
  AUBIO_FREE(c);
}

void aubio_batchnorm_debug(aubio_batchnorm_t *c, aubio_tensor_t *input_tensor)
{
  AUBIO_DBG("batchnorm: %15s -> %s (%d params) (4 * (%d,))\n",
      aubio_tensor_get_shape_string(input_tensor),
      aubio_tensor_get_shape_string(input_tensor), // same output shape
      c->n_outputs, 4 * c->n_outputs);
}

uint_t aubio_batchnorm_get_output_shape(aubio_batchnorm_t *c,
    aubio_tensor_t *input, uint_t *shape)
{
  AUBIO_ASSERT(c && input && shape);

  shape[0] = input->shape[0];
  shape[1] = input->shape[1];
  shape[2] = input->shape[2];

  aubio_batchnorm_debug(c, input);

  return AUBIO_OK;
}

void aubio_batchnorm_do(aubio_batchnorm_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  uint_t i, j, k;
  uint_t jj;
  smpl_t s;
  AUBIO_ASSERT(c);
  AUBIO_ASSERT_EQUAL_SHAPE(input_tensor, activations);
  if (input_tensor->ndim == 3) {
    for (i = 0; i < activations->shape[0]; i++) {
      jj = 0;
      for (j = 0; j < activations->shape[1]; j++) {
        for (k = 0; k < activations->shape[2]; k++) {
          s = input_tensor->data[i][jj + k];
          s -= c->moving_mean->data[k];
          s *= c->gamma->data[k];
          s /= SQRT(c->moving_variance->data[k] + 1.e-4);
          s += c->beta->data[k];
          activations->data[i][jj + k] = s;
        }
        jj += activations->shape[2];
      }
    }
  } else if (input_tensor->ndim == 2) {
    for (i = 0; i < activations->shape[0]; i++) {
      for (j = 0; j < activations->shape[1]; j++) {
        s = input_tensor->data[i][j];
        s -= c->moving_mean->data[j];
        s *= c->gamma->data[j];
        s /= SQRT(c->moving_variance->data[j] + 1.e-4);
        s += c->beta->data[j];
        activations->data[i][j] = s;
      }
    }
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
  AUBIO_ASSERT(t && t->beta);
  AUBIO_ASSERT(beta);
  if (t->beta->length != beta->length) return AUBIO_FAIL;
  fvec_copy(beta, t->beta);
  return AUBIO_OK;
}

uint_t aubio_batchnorm_set_moving_mean(aubio_batchnorm_t *t, fvec_t *moving_mean)
{
  AUBIO_ASSERT(t && t->moving_mean);
  AUBIO_ASSERT(moving_mean);
  if (t->moving_mean->length != moving_mean->length) return AUBIO_FAIL;
  fvec_copy(moving_mean, t->moving_mean);
  return AUBIO_OK;
}

uint_t aubio_batchnorm_set_moving_variance(aubio_batchnorm_t *t, fvec_t *moving_variance)
{
  AUBIO_ASSERT(t && t->moving_variance);
  AUBIO_ASSERT(moving_variance);
  if (t->moving_variance->length != moving_variance->length) return AUBIO_FAIL;
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

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
#include "dense.h"

struct _aubio_dense_t {
  uint_t n_units;
  fmat_t *weights;
  fvec_t *bias;
};

aubio_dense_t *new_aubio_dense(uint_t n_units) {
  aubio_dense_t *c = AUBIO_NEW(aubio_dense_t);

  AUBIO_GOTO_FAILURE((sint_t)n_units >= 1);

  c->n_units = n_units;

  return c;
failure:
  del_aubio_dense(c);
  return NULL;
}

void del_aubio_dense(aubio_dense_t *c) {
  AUBIO_ASSERT(c);
  if (c->weights)
    del_fmat(c->weights);
  if (c->bias)
    del_fvec(c->bias);
  AUBIO_FREE(c);
}

void aubio_dense_debug(aubio_dense_t *c, aubio_tensor_t *input_tensor)
{
  char_t input_string[15];
  snprintf(input_string, 15, "(%d)", input_tensor->shape[0]);
  AUBIO_DBG("dense:     %15s ¤ (%d, %d) ->"
      " (%d) (%d params)\n",
      input_string,
      c->n_units,
      c->n_units,
      input_tensor->shape[0] * c->n_units);
}

uint_t aubio_dense_get_output_shape(aubio_dense_t *c,
    aubio_tensor_t *input, uint_t *shape)
{
  AUBIO_ASSERT (c && input && shape);
  AUBIO_ASSERT (input->ndim == 1);
  shape[0] = c->n_units;

  if (c->weights) del_fmat(c->weights);
  c->weights = new_fmat(input->shape[0], c->n_units);
  if (!c->weights) return AUBIO_FAIL;

  if (c->bias) del_fvec(c->bias);
  c->bias = new_fvec(c->n_units);
  if (!c->bias) return AUBIO_FAIL;

  aubio_dense_debug(c, input);

  return AUBIO_OK;
}

fmat_t *aubio_dense_get_weights(aubio_dense_t *c) {
  return c->weights;
}

fvec_t *aubio_dense_get_bias(aubio_dense_t *c) {
  return c->bias;
}

void aubio_dense_do(aubio_dense_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations) {
  AUBIO_ASSERT(c && input_tensor && activations);
  AUBIO_ASSERT(input_tensor->ndim == 1);
  AUBIO_ASSERT(activations->ndim == 1);
  AUBIO_ASSERT(input_tensor->shape[0] == c->weights->height);
  AUBIO_ASSERT(activations->shape[0] == c->weights->length);

  fvec_t input_vec;
  aubio_tensor_as_fvec(input_tensor, &input_vec);
  fvec_t output_vec;
  aubio_tensor_as_fvec(activations, &output_vec);

  // compute x.W
  fvec_matmul(&input_vec, c->weights, &output_vec);
  // add bias
  fvec_vecadd(&output_vec, c->bias);

  // compute sigmoid
  uint_t i;
  for (i = 0; i < output_vec.length; i++) {
    output_vec.data[i] = 1. / (1. + EXP( - output_vec.data[i] ));
  }
}

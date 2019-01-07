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
#include "maxpool1d.h"

#include <float.h> // FLT_MAX

struct _aubio_maxpool1d_t {
  uint_t pool_size;
  uint_t stride;
};

static void aubio_maxpool1d_debug(aubio_maxpool1d_t *c,
    aubio_tensor_t *input_tensor);

aubio_maxpool1d_t *new_aubio_maxpool1d(uint_t pool_size[1])
{
  aubio_maxpool1d_t *c = AUBIO_NEW(aubio_maxpool1d_t);

  AUBIO_GOTO_FAILURE((sint_t)pool_size[0] > 0);

  c->pool_size = pool_size[0];

  c->stride = 1;

  return c;

failure:
  del_aubio_maxpool1d(c);
  return NULL;
}

void del_aubio_maxpool1d(aubio_maxpool1d_t* c) {
  AUBIO_ASSERT(c);
  AUBIO_FREE(c);
}

void aubio_maxpool1d_debug(aubio_maxpool1d_t *c, aubio_tensor_t *input_tensor)
{
  AUBIO_DBG("maxpool1d: input (%d, %d) ¤ maxpool1d (pool_size = (%d)) ->"
      " (%d, %d) (no params)\n",
      input_tensor->shape[0],
      input_tensor->shape[1],
      c->pool_size,
      input_tensor->shape[0] / c->pool_size,
      input_tensor->shape[1]);
}

uint_t aubio_maxpool1d_get_output_shape(aubio_maxpool1d_t *c,
    aubio_tensor_t *input, uint_t *shape)
{
  AUBIO_ASSERT(c);
  AUBIO_ASSERT(shape && sizeof(shape) == 2*sizeof(uint_t));
  AUBIO_ASSERT(input);
  shape[0] = input->shape[0] / c->pool_size;
  shape[1] = input->shape[1];

  aubio_maxpool1d_debug(c, input);

  return AUBIO_OK;
}

void aubio_maxpool1d_do(aubio_maxpool1d_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *output_tensor)
{
  uint_t i, j, a;
  AUBIO_ASSERT(c && input_tensor && output_tensor);

  //aubio_maxpool1d_debug(c, input_tensor);

  for (j = 0; j < output_tensor->shape[1]; j++) {
    for (i = 0; i < output_tensor->shape[0]; i++) {
      uint_t stride_i = i * c->pool_size;
      smpl_t m = input_tensor->data[stride_i][j];
      for (a = 0; a < c->pool_size; a++) {
        m = MAX(m, input_tensor->data[stride_i + a][j]);
      }
      output_tensor->data[i][j] = m;
    }
  }
}

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
#include "conv1d.h"

typedef enum
{
  PAD_SAME = 0,
  PAD_VALID = 1,
  PAD_CAUSAL = 2, // TODO (1d only, for dilated convolution)
} aubio_conv1d_padding_type;

struct _aubio_conv1d_t {
  // define internals here
  uint_t n_filters;
  uint_t kernel_shape;     // kernel sizes
  uint_t stride_shape;     // stride sizes

  aubio_conv1d_padding_type padding_mode;

  // these will be set after calling get_output_shape
  aubio_tensor_t *kernel;
  fvec_t *bias;
  uint_t output_shape[2];  // shape of output
  uint_t padding_start;    // left padding

#if defined(HAVE_BLAS)
  aubio_tensor_t *padded_input;
#endif
};

static void aubio_conv1d_debug(aubio_conv1d_t *c, aubio_tensor_t *input_tensor);

aubio_conv1d_t *new_aubio_conv1d(uint_t n_filters, uint_t kernel_shape[1])
{
  aubio_conv1d_t *c = AUBIO_NEW(aubio_conv1d_t);

  // validate input parameters
  AUBIO_GOTO_FAILURE((sint_t)n_filters >= 1);
  AUBIO_GOTO_FAILURE((sint_t)kernel_shape[0] >= 1);

  // set internal variables
  c->n_filters = n_filters;
  c->kernel_shape = kernel_shape[0];

  // default to padding_mode="valid"
  c->padding_mode = PAD_VALID;
  // set default stride_shape to (1)
  uint_t stride_shape[1] = {1};
  aubio_conv1d_set_stride(c, stride_shape);

  return c;

failure:
  del_aubio_conv1d(c);
  return NULL;
}

void del_aubio_conv1d(aubio_conv1d_t *c)
{
  AUBIO_ASSERT(c);
  // destroy internals here
  if (c->kernel) {
    del_aubio_tensor(c->kernel);
  }
  if (c->bias)
    del_fvec(c->bias);
#if defined(HAVE_BLAS)
  if (c->padded_input) del_aubio_tensor(c->padded_input);
#endif
  AUBIO_FREE(c);
}


uint_t aubio_conv1d_set_stride(aubio_conv1d_t *c, uint_t stride[1])
{
  if ((sint_t)stride[0] < 1) return AUBIO_FAIL;
  c->stride_shape = stride[0];
  return AUBIO_OK;
}

uint_t aubio_conv1d_get_stride(aubio_conv1d_t *c)
{
  return c->stride_shape;
}

uint_t aubio_conv1d_get_output_shape(aubio_conv1d_t *c,
    aubio_tensor_t *input_tensor,
    uint_t *shape)
{
  uint_t output_shape[2] = {0, c->n_filters};
  uint_t padding_shape = 0;  // total amount of padding
  uint_t padding_start = 0;

  // check input parameters
  AUBIO_ASSERT(input_tensor);
  AUBIO_ASSERT(shape);

  // reset output array
  shape[0] = 0;
  shape[1] = 0;

  switch (c->padding_mode) {
    case PAD_SAME:
      // compute output shape
      output_shape[0] = (uint_t)CEIL(input_tensor->shape[0]
          / (smpl_t)c->stride_shape);

      padding_shape = (output_shape[0] - 1) * c->stride_shape +
        c->kernel_shape - input_tensor->shape[0];

      padding_start = FLOOR(padding_shape / 2);
      break;
    case PAD_VALID:
      output_shape[0] = (input_tensor->shape[0] - c->kernel_shape + 1)
        / c->stride_shape;

      padding_start = 0;
      break;
    case PAD_CAUSAL:
      // TODO
      return AUBIO_FAIL;
    default:
      return AUBIO_FAIL;
  }

  uint_t kernel_shape[3];
  kernel_shape[0] = c->kernel_shape; // filter length
  kernel_shape[1] = input_tensor->shape[1]; // channels
  kernel_shape[2] = c->n_filters; // outputs

  if (c->kernel) del_aubio_tensor(c->kernel);
  if (c->bias) del_fvec(c->bias);

  c->kernel = new_aubio_tensor(3, kernel_shape);
  if (!c->kernel) return AUBIO_FAIL;
  c->bias = new_fvec(c->n_filters);

  // set internals upon success
  c->output_shape[0] = output_shape[0];
  c->output_shape[1] = output_shape[1];

#if defined(HAVE_BLAS)
  if (c->padded_input) del_aubio_tensor(c->padded_input);
  uint_t padded_shape[2] = {input_tensor->shape[0] + padding_shape,
    input_tensor->shape[1]};
  c->padded_input = new_aubio_tensor(2, padded_shape);
#endif

  c->padding_start = padding_start;

  // set output
  shape[0] = output_shape[0];
  shape[1] = output_shape[1];

  aubio_conv1d_debug(c, input_tensor);

  return AUBIO_OK;
}

void aubio_conv1d_debug(aubio_conv1d_t *c, aubio_tensor_t *input_tensor)
{
  // print some info
  AUBIO_ASSERT(c);
  uint_t n_params = (c->kernel->shape[0] * c->kernel->shape[2] + 1)
    * c->kernel->shape[1];
  AUBIO_DBG("conv1d:    %15s -> (%d, %d) (%d params)"
      " (weigths=(%d, %d, %d), stride=(%d,), pad_start=(%d,))\n",
    aubio_tensor_get_shape_string(input_tensor),
    c->output_shape[0], c->output_shape[1],
    n_params,
    c->kernel->shape[0], c->kernel->shape[1], c->kernel->shape[2],
    c->stride_shape,
    -c->padding_start);
}

uint_t aubio_conv1d_check_output_shape(aubio_conv1d_t *c,
    aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  // fetch output_shape if it hasn't been done before
  if (c->output_shape[0] == 0 ||
      c->output_shape[1] == 0) {
    if (!aubio_conv1d_get_output_shape(c, input_tensor, c->output_shape)) {
      return AUBIO_FAIL;
    }
  }

  // check we have as many filters as expected activation outputs
  if (activations->shape[1] != c->n_filters) return AUBIO_FAIL;
  if (activations->shape[1] != c->kernel->shape[2]) return AUBIO_FAIL;
  if (input_tensor->shape[1] != c->kernel->shape[1]) return AUBIO_FAIL;

  // check tensor activations has the expected sizes
  if (c->output_shape[0] != activations->shape[0]) return AUBIO_FAIL;
  if (c->output_shape[1] != activations->shape[1]) return AUBIO_FAIL;
  return AUBIO_OK;
}

#if !defined(HAVE_BLAS)
void aubio_conv1d_do(aubio_conv1d_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  uint_t i, j, k, a;
  uint_t stride_a, kk;
  sint_t x;
  smpl_t s, w, bias, acc;

  AUBIO_ASSERT(c && input_tensor && activations);
  // check we have the correct output activation sizes
  if (aubio_conv1d_check_output_shape(c, input_tensor, activations))
  {
    AUBIO_ERR("conv1d: check_output_shape failed\n");
    return;
  }

  // for each kernel filter k
  for (i = 0; i < activations->shape[1]; i++) {
    // get bias
    bias = c->bias->data[i];
    stride_a = 0; // k * c->stride_shape
    // for each output
    for (j = 0; j < activations->shape[0]; j++) {
      // reset output
      acc = 0;
      // compute convolution for one kernel
      for (a = 0; a < c->kernel_shape; a++) {
        x = stride_a + a - c->padding_start;
        if ((x > -1) && (x < (sint_t)input_tensor->shape[0])) {
          kk = 0;
          // for each input channel
          for (k = 0; k < input_tensor->shape[1]; k++) {
            // get kernel weight
            w = c->kernel->data[a][kk + i];
            // get input sample
            s = input_tensor->data[x][k];
            acc += w * s;
            kk += c->kernel->shape[2];
          }
        }
      }
      stride_a += c->stride_shape;
      // apply bias
      acc += bias;
      // compute RELU
      activations->data[j][i] = MAX(acc, 0);
    }
  }
}

#else /* HAVE_BLAS */

// blas implementation
//
//  uses gemv on the padded input to compute each output elements at once
//
// TODO
//  - avoid copy when padding_start == 0
//  - optimize copying using tensor helpers

void aubio_conv1d_do(aubio_conv1d_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  uint_t i, j;

  uint_t sdot_size = c->kernel->shape[0] * c->kernel->shape[1];
  uint_t input_stride = c->stride_shape * c->padded_input->shape[1];

  AUBIO_ASSERT(c && input_tensor && activations);
  if (aubio_conv1d_check_output_shape(c, input_tensor, activations))
  {
    AUBIO_ERR("conv1d: check_output_shape failed\n");
    return;
  }

  // copy input to padded version
  for (j = 0; j < input_tensor->shape[0]; j++) {
    for (i = 0; i < input_tensor->shape[1]; i++) {
      c->padded_input->data[j + c->padding_start][i] =
        input_tensor->data[j][i];
    }
  }

  // for each output
  for (j = 0; j < activations->shape[0]; j++) {
    // for each row of activation output
    aubio_cblas__gemv(CblasRowMajor, CblasTrans,
        sdot_size, c->kernel->shape[2], 1.,
        c->kernel->buffer, c->kernel->shape[2],
        c->padded_input->buffer + j  * input_stride, 1, 0.,
        activations->buffer + j * activations->shape[1], 1);
  }
  for (j = 0; j < activations->shape[0]; j++) {
    // for each kernel filter k
    for (i = 0; i < activations->shape[1]; i++) {
      activations->data[j][i] += c->bias->data[i];
      activations->data[j][i] = MAX(activations->data[j][i], 0);
    }
  }
}
#endif /* HAVE_BLAS */

uint_t aubio_conv1d_set_padding_mode(aubio_conv1d_t *c,
    const char_t *padding_mode)
{
  AUBIO_ASSERT(c && padding_mode);
  if (strncmp(padding_mode, "same", PATH_MAX) == 0) {
    c->padding_mode = PAD_SAME;
  } else if (strncmp(padding_mode, "valid", PATH_MAX) == 0) {
    c->padding_mode = PAD_VALID;
  } else {
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

aubio_tensor_t *aubio_conv1d_get_kernel(aubio_conv1d_t* c)
{
  AUBIO_ASSERT(c && c->kernel);
  return c->kernel;
}

fvec_t *aubio_conv1d_get_bias(aubio_conv1d_t* c)
{
  AUBIO_ASSERT(c && c->bias);
  return c->bias;
}

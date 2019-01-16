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
#include "conv2d.h"

typedef enum
{
  PAD_SAME = 0,   // same, aka half mode
  PAD_VALID = 1   // valid, aka no padding
} aubio_conv2d_padding_t;

struct _aubio_conv2d_t {
  // define internals here
  uint_t n_filters;
  uint_t kernel_shape[2];     // kernel sizes
  uint_t stride_shape[2];     // stride sizes

  aubio_conv2d_padding_t padding_mode;

  // these will be set after calling get_output_shape
  aubio_tensor_t *kernel;
  fvec_t *bias;
  uint_t output_shape[3];     // shape of output
  uint_t padding_start[2];    // {top, left} padding

#if defined(HAVE_BLAS)
  aubio_tensor_t *padded_input;
#endif
};

static void aubio_conv2d_debug(aubio_conv2d_t *c, aubio_tensor_t *input_tensor);

aubio_conv2d_t *new_aubio_conv2d(uint_t n_filters, uint_t *kernel_shape)
{
  aubio_conv2d_t *c = AUBIO_NEW(aubio_conv2d_t);

  // validate input parameters
  AUBIO_GOTO_FAILURE((sint_t)n_filters >= 1);
  AUBIO_GOTO_FAILURE((sint_t)kernel_shape[0] >= 1);
  AUBIO_GOTO_FAILURE((sint_t)kernel_shape[1] >= 1);

  // set internal variables
  c->n_filters = n_filters;
  c->kernel_shape[0] = kernel_shape[0];
  c->kernel_shape[1] = kernel_shape[1];

  // default to padding_mode="valid"
  c->padding_mode = PAD_VALID;
  // set default stride_shape to {1, 1}
  {
    uint_t default_stride[2] = {1, 1};
    aubio_conv2d_set_stride(c, default_stride);
  }

  return c;

failure:
  del_aubio_conv2d(c);
  return NULL;
}

void del_aubio_conv2d(aubio_conv2d_t *c)
{
  AUBIO_ASSERT(c);
  if (c->kernel)
    del_aubio_tensor(c->kernel);
  if (c->bias)
    del_fvec(c->bias);
#if defined(HAVE_BLAS)
  if (c->padded_input)
    del_aubio_tensor(c->padded_input);
#endif
  AUBIO_FREE(c);
}


uint_t aubio_conv2d_set_stride(aubio_conv2d_t *c,
    uint_t stride[2])
{
  if ((sint_t)stride[0] < 1) return AUBIO_FAIL;
  if ((sint_t)stride[1] < 1) return AUBIO_FAIL;
  c->stride_shape[0] = stride[0];
  c->stride_shape[1] = stride[1];
  return AUBIO_OK;
}

uint_t *aubio_conv2d_get_stride(aubio_conv2d_t *c)
{
  return c->stride_shape;
}

uint_t aubio_conv2d_get_output_shape(aubio_conv2d_t *c,
    aubio_tensor_t *input_tensor,
    uint_t *shape)
{
  uint_t output_shape[3] = {0, 0, c->n_filters};
  uint_t padding_start[2] = {0, 0};
  // total amount of padding
  uint_t padding_shape[2] = {0, 0};

  // check input parameters
  AUBIO_ASSERT(input_tensor);
  AUBIO_ASSERT(shape);

  // reset output array
  shape[0] = 0;
  shape[1] = 0;
  shape[2] = 0;

  switch (c->padding_mode) {
    case PAD_SAME:
      // compute output shape
      output_shape[0] = (uint_t)CEIL(input_tensor->shape[0]
          / (smpl_t)c->stride_shape[0]);
      output_shape[1] = (uint_t)CEIL(input_tensor->shape[1]
          / (smpl_t)c->stride_shape[1]);

      padding_shape[0] = (output_shape[0] - 1) * c->stride_shape[0]
        + c->kernel_shape[0] - input_tensor->shape[0];
      padding_shape[1] = (output_shape[1] - 1) * c->stride_shape[1]
        + c->kernel_shape[1] - input_tensor->shape[1];

      padding_start[0] = FLOOR(padding_shape[0] / 2);
      padding_start[1] = FLOOR(padding_shape[1] / 2);

      break;
    case PAD_VALID:
      output_shape[0] = (input_tensor->shape[0] - c->kernel_shape[0] + 1)
        / c->stride_shape[0];
      output_shape[1] = (input_tensor->shape[1] - c->kernel_shape[1] + 1)
        / c->stride_shape[1];

      padding_start[0] = 0;
      padding_start[1] = 0;

      break;
    //case PAD_CAUSAL:
    //  // TODO
    //  return AUBIO_FAIL;
    default:
      return AUBIO_FAIL;
  }

  uint_t kernel_shape[4];
  kernel_shape[0] = c->kernel_shape[0];
  kernel_shape[1] = c->kernel_shape[1];
  kernel_shape[2] = input_tensor->shape[2];
  kernel_shape[3] = c->n_filters;

  if (c->kernel) del_aubio_tensor(c->kernel);
  if (c->bias) del_fvec(c->bias);

  c->kernel = new_aubio_tensor(4, kernel_shape);
  if (!c->kernel) return AUBIO_FAIL;
  c->bias = new_fvec(c->n_filters);

  // set internals upon success
  c->output_shape[0] = output_shape[0];
  c->output_shape[1] = output_shape[1];
  c->output_shape[2] = output_shape[2];

  c->padding_start[0] = padding_start[0];
  c->padding_start[1] = padding_start[1];

  // set output
  shape[0] = output_shape[0];
  shape[1] = output_shape[1];
  shape[2] = output_shape[2];


#if defined(HAVE_BLAS)
  // im2col padding
  padding_shape[0] = output_shape[0] * output_shape[1];
  padding_shape[1] = c->kernel_shape[0] * c->kernel_shape[1]
    * input_tensor->shape[2];
  c->padded_input = new_aubio_tensor(2, padding_shape);
  if (!c-> padded_input) {
    AUBIO_MSG("conv2d: failed creating padded_input with shape (%d, %d, %d)\n",
        padding_shape);
    return AUBIO_FAIL;
  }
#endif

  aubio_conv2d_debug(c, input_tensor);

  return AUBIO_OK;
}

void aubio_conv2d_debug(aubio_conv2d_t *c, aubio_tensor_t *input_tensor)
{
  // print some info
  AUBIO_ASSERT(c);
  uint_t n_params = (c->kernel->shape[0] * c->kernel->shape[2] + 1)
    * c->kernel->shape[1] * c->kernel->shape[3];

  const char_t *tensor_str = aubio_tensor_get_shape_string(input_tensor);
  //AUBIO_DBG("conv2d: kernel_shape_str %s\n", kernel_shape_str);
  AUBIO_DBG("conv2d:    %15s -> (%d, %d, %d)",
    tensor_str,
    c->output_shape[0], c->output_shape[1], c->output_shape[2]);
  tensor_str = aubio_tensor_get_shape_string(c->kernel);
  AUBIO_DBG(" (n_params=%d, kernel_shape=(%d, %d),"
      " weigths=%s, stride (%d, %d), pad_start [%d, %d])\n",
    n_params, c->kernel_shape[0], c->kernel_shape[1],
    tensor_str,
    c->stride_shape[0], c->stride_shape[1],
    -c->padding_start[0], -c->padding_start[1]);
}

uint_t aubio_conv2d_check_output_shape(aubio_conv2d_t *c,
    aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  // fetch output_shape if it hasn't been done before
  if (c->output_shape[0] == 0 ||
      c->output_shape[1] == 0 ||
      c->output_shape[2] == 0) {
    if (!aubio_conv2d_get_output_shape(c, input_tensor, c->output_shape)) {
      return AUBIO_FAIL;
    }
  }

  // check we have as many filters as expected activation outputs
  if (activations->shape[2] != c->n_filters) return AUBIO_FAIL;
  if (activations->shape[2] != c->kernel->shape[3]) return AUBIO_FAIL;
  if (input_tensor->shape[2] != c->kernel->shape[2]) return AUBIO_FAIL;

  // check tensor activations has the expected sizes
  if (c->output_shape[0] != activations->shape[0]) return AUBIO_FAIL;
  if (c->output_shape[1] != activations->shape[1]) return AUBIO_FAIL;
  if (c->output_shape[2] != activations->shape[2]) return AUBIO_FAIL;
  return AUBIO_OK;
}

#if !defined(HAVE_BLAS)
void aubio_conv2d_do(aubio_conv2d_t *c, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  uint_t i, j, k, l, a, b;
  uint_t stride_a, stride_b;
  sint_t x, y;
  smpl_t s, w, bias, acc;
  uint_t jj, ll, bb, yy;

  uint_t k_stride1 = c->kernel->shape[3];
  uint_t k_stride2 = c->kernel->shape[2] * k_stride1;

  AUBIO_ASSERT(c && input_tensor && activations);
  // check we have the correct output activation sizes
  if (aubio_conv2d_check_output_shape(c, input_tensor, activations))
  {
    AUBIO_ERR("conv2d: check_output_shape failed\n");
    return;
  }

  // for each kernel filter k
  for (i = 0; i < activations->shape[2]; i++) {
    // get bias
    bias = c->bias->data[i];
    stride_b = 0; // == j * c->stride_shape[1]
    jj = 0; // == j * activations->shape[2]
    // for each output y
    for (j = 0; j < activations->shape[1]; j++) {
      // for each output x
      stride_a = 0; // k * c->stride_shape[0]
      for (k = 0; k < activations->shape[0]; k++) {
        // reset output
        acc = 0;
        // compute convolution for one kernel
        for (a = 0; a < c->kernel_shape[0]; a++) {
          x = stride_a + a - c->padding_start[0];
          if ((x < 0) || (x > (sint_t)input_tensor->shape[0] - 1))
            continue; // padding with 0.
          bb = 0; // == b * k_stride2
          for (b = 0; b < c->kernel_shape[1]; b++) {
            y = stride_b + b - c->padding_start[1];
            if ((y < 0) || (y > (sint_t)input_tensor->shape[1] - 1))
              continue; // padding with 0.
            yy = y * input_tensor->shape[2];
            ll = bb + i; // + l * k_stride1
            // for each input channel
            for (l = 0; l < input_tensor->shape[2]; l++) {
              // get kernel weight
              w = c->kernel->data[a][ll];
              // get input sample
              s = input_tensor->data[x][yy + l];
              acc += w * s;
              ll += k_stride1;
            }
            bb += k_stride2;
          }
        }
        stride_a += c->stride_shape[0];
        // apply bias
        acc += bias;
        // set output activation
        activations->data[k][jj + i] = acc;
      }
      stride_b += c->stride_shape[1];
      jj += activations->shape[2];
    }
  }
}

#else /* HAVE_BLAS */

void aubio_conv2d_copy_to_padded(aubio_conv2d_t *o,
    aubio_tensor_t *input_tensor, aubio_tensor_t *padded_input)
{
  // naive implementation of im2col
  uint_t i, j, k, l, m;
  uint_t stride_4 = o->kernel->shape[2];
  uint_t stride_3 = o->kernel->shape[1] * stride_4;
  uint_t stride_2 = o->kernel->shape[0] * stride_3;
  uint_t stride_1 = o->output_shape[1] * stride_2;
  uint_t stride_in_2 = input_tensor->shape[2];
  uint_t stride_in_1 = input_tensor->shape[1] * stride_in_2;

  AUBIO_ASSERT(padded_input->size ==
      o->output_shape[0] * o->output_shape[1]
      * o->kernel_shape[0] * o->kernel_shape[1]
      * input_tensor->shape[2]);
  AUBIO_ASSERT(input_tensor->shape[2] == o->kernel->shape[2]);

  for (i = 0; i < o->output_shape[0]; i++)
  {
    for (j = 0; j <  o->output_shape[1]; j++)
    {
      for (k = 0; k < o->kernel->shape[0]; k++)
      {
        for (l = 0; l < o->kernel->shape[1]; l++)
        {
          for (m = 0; m < o->kernel->shape[2]; m++)
          {
            uint_t read_i = i * o->stride_shape[0] + k;
            uint_t read_j = j * o->stride_shape[1] + l;
            if (read_i < o->padding_start[0])
              continue;
            else if (read_i - o->padding_start[0] >= input_tensor->shape[0])
              continue;
            if (read_j < o->padding_start[1])
              continue;
            else if (read_j - o->padding_start[1] >= input_tensor->shape[1])
              continue;

            sint_t idx =
              ((read_i - o->padding_start[0])) * stride_in_1
              + ((read_j - o->padding_start[1])) * stride_in_2
              + m;
            padded_input->buffer[i * stride_1
              + j * stride_2
              + k * stride_3
              + l * stride_4
              + m]
              = input_tensor->buffer[idx];
          }
        }
      }
    }
  }
}

void aubio_conv2d_do(aubio_conv2d_t *o, aubio_tensor_t *input_tensor,
    aubio_tensor_t *activations)
{
  uint_t i, j;
  smpl_t bias;
  aubio_tensor_t *padded_input = o->padded_input;
  aubio_tensor_t *kernel = o->kernel;

  AUBIO_ASSERT(o && input_tensor && activations);
  // check we have the correct output activation sizes
  if (aubio_conv2d_check_output_shape(o, input_tensor, activations))
  {
    AUBIO_ERR("conv2d: check_output_shape failed\n");
    return;
  }

  uint_t M = padded_input->shape[0];
  uint_t K = padded_input->size/padded_input->shape[0];
  uint_t N = kernel->size / K;

  // check sizes
  AUBIO_ASSERT(M * K == padded_input->size);
  AUBIO_ASSERT(N * K == kernel->size);
  AUBIO_ASSERT(M * N == activations->size);

  // copy input to im2col sliding window version
  aubio_conv2d_copy_to_padded(o, input_tensor, padded_input);

  aubio_cblas__gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      M,                    // M
      N,                    // N
      K,                    // K
      1.F,                  // alpha
      padded_input->buffer, // M x K matrix
      K,                    // K (2nd dim of A)
      kernel->buffer,       // K x N matrix
      N,                    // N
      0.F,                  // beta
      activations->buffer,  // M x N matrix
      N);                   // N (2nd dim of C)


  // apply bias
  for (i = 0; i < activations->shape[2]; i++) {
    bias = o->bias->data[i];
    for (j = 0; j < activations->shape[0] * activations->shape[1]; j++)
    {
      activations->buffer[j * activations->shape[2] + i] += bias;
    }
  }
}
#endif

void aubio_conv2d_do_backwards(aubio_conv2d_t *c,
    /*aubio_tensor_t *old_gradients,*/
    aubio_tensor_t *gradients)
{
  uint_t i, j, k, a, b;
  AUBIO_ASSERT(c && gradients);
  // TODO
  // for each kernel filter k
  for (i = 0; i < c->n_filters; i++) {
    // for each input column
    for (j = 0; j < gradients->shape[1]; j++) {
      // for each input row
      for (k = 0; k < gradients->shape[2]; k++) {
        for (a = 0; a < c->kernel_shape[0]; a++) {
          for (b = 0; b < c->kernel_shape[1]; b++) {
#if 0
            smpl_t grad = gradients->data[i]->data[a][b];
            smpl_t oldgrad = old_gradients->data[i]->data[a][b];
            smpl_t m = (grad - oldgrad * momentum);
            w -= lr * m - lr * decay * w;
#endif
          }
        }
      }
    }
  }
}

uint_t aubio_conv2d_set_padding_mode(aubio_conv2d_t *c,
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

uint_t aubio_conv2d_set_kernel(aubio_conv2d_t *c, aubio_tensor_t *kernel)
{
  uint_t i;
  AUBIO_ASSERT(c && kernel);
  for (i = 0; i < c->kernel->ndim; i++) {
    AUBIO_ASSERT(c->kernel->shape[i] == kernel->shape[i]);
  }
  return AUBIO_OK;
}

aubio_tensor_t *aubio_conv2d_get_kernel(aubio_conv2d_t* c)
{
  AUBIO_ASSERT(c && c->kernel);
  return c->kernel;
}

uint_t aubio_conv2d_set_bias(aubio_conv2d_t *c, fvec_t *bias)
{
  AUBIO_ASSERT(c && bias);
  AUBIO_ASSERT(c->kernel_shape[1] == bias->length);
  return AUBIO_OK;
}

fvec_t *aubio_conv2d_get_bias(aubio_conv2d_t* c)
{
  AUBIO_ASSERT(c && c->bias);
  return c->bias;
}

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

#ifndef AUBIO_CONV2D_H
#define AUBIO_CONV2D_H

/** \file

  Convolutional layer (2D)

  Standard implementation of a 2D convolutional layer. partly optimized for
  CPU.

  Note
  ----
  Only the forward pass is implemented for now.

  References
  ----------
  Vincent Dumoulin, Francesco Visin - [A guide to convolution arithmetic for
  deep learning](https://github.com/vdumoulin/conv_arithmetic)

*/

#ifdef __cplusplus
extern "C" {
#endif

/** conv2d layer */
typedef struct _aubio_conv2d_t aubio_conv2d_t;

/** create a new conv2d layer

  \param n_filters number of filters
  \param kernel_shape shape of each filter

  \return new conv2d layer

*/
aubio_conv2d_t *new_aubio_conv2d(uint_t n_filters, uint_t kernel_shape[2]);

/** set padding mode

  \param c                  layer
  \param padding_mode       padding mode

  \return 0 on success, non-zero otherwise

  Available padding: "same", and "valid".

*/
uint_t aubio_conv2d_set_padding_mode(aubio_conv2d_t *c,
    const char_t *padding_mode);

/** set stride

  \param c      layer
  \param stride array of length 2 containing the strides

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_conv2d_set_stride(aubio_conv2d_t *c, uint_t stride[2]);

/** get current stride settings

  \param t  layer

  \return   array of length 2 containing the stride in each dimension

*/
uint_t *aubio_conv2d_get_stride(aubio_conv2d_t* t);

/** get output shape

  \param t                  layer
  \param input_tensor       input tensor
  \param shape              output shape

  \return 0 on success, non-zero otherwise

  Upon return, `shape` will be filled with the output shape of the layer. This
  function should be called after ::aubio_conv2d_set_stride or
  ::aubio_conv2d_set_padding_mode, and before ::aubio_conv2d_get_kernel or
  ::aubio_conv2d_get_bias.

*/
uint_t aubio_conv2d_get_output_shape(aubio_conv2d_t *t,
        aubio_tensor_t *input_tensor, uint_t *shape);

/** get kernel weights

  \param t  ::aubio_conv2d_t layer

  \return tensor of weights

  When called after ::aubio_conv2d_get_output_shape, this function will return
  a pointer to the tensor holding the weights of this layer.

*/
aubio_tensor_t *aubio_conv2d_get_kernel(aubio_conv2d_t *t);

/** get biases

  \param t layer

  \return vector of biases

  When called after ::aubio_conv2d_get_output_shape, this function will return
  a pointer to the vector holding the biases.

*/
fvec_t *aubio_conv2d_get_bias(aubio_conv2d_t *t);

/** set kernel weights

  \param t          layer
  \param kernel     kernel weights

  \return 0 on success, non-zero otherwise

  Copy kernel weights into internal layer memory.  This function should be
  called after ::aubio_conv2d_get_output_shape.

*/
uint_t aubio_conv2d_set_kernel(aubio_conv2d_t *t, aubio_tensor_t *kernel);

/** set biases

  \param t          layer
  \param bias       biases

  \return 0 on success, non-zero otherwise

  Copy vector of biases into internal layer memory. This function should be
  called after ::aubio_conv2d_get_output_shape.

*/
uint_t aubio_conv2d_set_bias(aubio_conv2d_t *t, fvec_t *bias);

/** compute layer output

  \param t              layer
  \param input_tensor   input tensor
  \param output_tensor  output tensor

  Perform 2D convolution.

*/
void aubio_conv2d_do(aubio_conv2d_t *t, aubio_tensor_t *input_tensor,
        aubio_tensor_t *output_tensor);

/** destroy conv2d layer

  \param t  layer

*/
void del_aubio_conv2d(aubio_conv2d_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_CONV2D_H */

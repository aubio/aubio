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

#ifndef AUBIO_CONV1D_H
#define AUBIO_CONV1D_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_conv1d_t aubio_conv1d_t;

/** create a new conv1d layer */
aubio_conv1d_t *new_aubio_conv1d(uint_t filters, uint_t kernel_shape[1]);

/** perform forward 1D convolution */
void aubio_conv1d_do(aubio_conv1d_t *t, aubio_tensor_t *input_tensor,
        aubio_tensor_t *activations);

/** TODO: implement */
void aubio_conv1d_train(aubio_conv1d_t *t, aubio_tensor_t *input_tensor);

/** get conv1d weights */
aubio_tensor_t *aubio_conv1d_get_kernel(aubio_conv1d_t *t);

/** get conv1d biases */
fvec_t *aubio_conv1d_get_bias(aubio_conv1d_t *t);

/** set conv1d stride */
uint_t aubio_conv1d_set_stride(aubio_conv1d_t *c, uint_t stride[1]);

uint_t aubio_conv1d_set_padding_mode(aubio_conv1d_t *c,
    const char_t *padding_mode);

uint_t aubio_conv1d_get_output_shape(aubio_conv1d_t *t,
        aubio_tensor_t *input_tensor, uint_t *shape);

void del_aubio_conv1d(aubio_conv1d_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_CONV1D_H */

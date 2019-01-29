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

#ifndef AUBIO_MAXPOOL2D_H
#define AUBIO_MAXPOOL2D_H

/** \file

  Max pooling layer (2D)

*/

#ifdef __cplusplus
extern "C" {
#endif

/** maxpool2d layer */
typedef struct _aubio_maxpool2d_t aubio_maxpool2d_t;

/** create a new maxpool2d layer

  \param pool_size  size of the pooling windows

  \return new ::aubio_maxpool2d_t layer

*/
aubio_maxpool2d_t *new_aubio_maxpool2d(uint_t pool_size[2]);

/** get output shape

  \param t      layer
  \param input  input tensor
  \param shape  output shape

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_maxpool2d_get_output_shape(aubio_maxpool2d_t *t,
        aubio_tensor_t *input, uint_t *shape);

/** compute layer output

  \param    t               layer
  \param    input_tensor    input tensor
  \param    output_tensor   output tensor

*/
void aubio_maxpool2d_do(aubio_maxpool2d_t *t,
        aubio_tensor_t *input_tensor,
        aubio_tensor_t *output_tensor);

/** destroy layer

  \param t      layer to destroy

*/
void del_aubio_maxpool2d(aubio_maxpool2d_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_MAXPOOL2D_H */

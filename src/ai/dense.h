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

#ifndef AUBIO_DENSE_H
#define AUBIO_DENSE_H

/** \file

 Fully connected layer

*/

#ifdef __cplusplus
extern "C" {
#endif

/** dense layer */
typedef struct _aubio_dense_t aubio_dense_t;

/** create a new dense layer

  \param n_units    number of units

  \return new dense layer

*/
aubio_dense_t *new_aubio_dense(uint_t n_units);

/** get output shape

  \param c      ::aubio_dense_t layer
  \param input  input tensor
  \param shape  output shape

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_dense_get_output_shape(aubio_dense_t *c,
    aubio_tensor_t *input, uint_t *shape);

/** get internal weights

   \param c     dense layer

   \return matrix of weights

   This function should be called after ::aubio_dense_get_output_shape
   to get a pointer to the internal weight matrix.

*/
fmat_t *aubio_dense_get_weights(aubio_dense_t *c);

/** get internal biases

   \param c     dense layer

   \return vector of biases

   This function should be called after ::aubio_dense_get_output_shape
   to get a pointer to the internal biases.

*/
fvec_t *aubio_dense_get_bias(aubio_dense_t *c);

/** compute forward pass

  \param c          ::aubio_dense_t layer
  \param input      input tensor
  \param output     output tensor

  This function computes the output of the dense layer given an input tensor.

*/
void aubio_dense_do(aubio_dense_t *c, aubio_tensor_t *input,
    aubio_tensor_t *output);

/** destroy layer

  \param c  layer to destroy

*/
void del_aubio_dense(aubio_dense_t *c);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_DENSE_H */

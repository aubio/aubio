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

#ifndef AUBIO_BATCHNORM_H
#define AUBIO_BATCHNORM_H

/** \file

 Batch normalization layer.

 References
 ----------

 Ioffe, Sergey; Szegedy, Christian. "Batch Normalization: Accelerating Deep
 Network Training by Reducing Internal Covariate Shift", available online
 at https://arxiv.org/pdf/1502.03167.pdf

*/

#ifdef __cplusplus
extern "C" {
#endif

/** batch normalization layer */
typedef struct _aubio_batchnorm_t aubio_batchnorm_t;

/** create a new batch normalization layer

  This layer takes no parameters. The number of output channels will be
  determined as the inner-most dimension of the input tensor when calling
  ::aubio_batchnorm_get_output_shape.

*/
aubio_batchnorm_t *new_aubio_batchnorm(void);

/** get output shape of the layer

  \param t      ::aubio_batchnorm_t layer
  \param input  input tensor
  \param shape  output shape

  This function determines the number of output channels required and allocate
  the vectors of weights. The ouptut shape of this layer is identical to the
  input shape.

*/
uint_t aubio_batchnorm_get_output_shape(aubio_batchnorm_t *t,
        aubio_tensor_t *input, uint_t *shape);

/** get a pointer to the gamma vector

  \param t  ::aubio_batchnorm_t layer

  \return   pointer to `fvec_t` holding the gamma parameters

  When called after ::aubio_batchnorm_get_output_shape, this function will
  return a pointer to the vector allocated to hold the `gamma` weights.

  A NULL pointer will be returned if ::aubio_batchnorm_get_output_shape has not
  been called yet.

*/
fvec_t *aubio_batchnorm_get_gamma(aubio_batchnorm_t *t);

/** get a pointer to the beta vector

  \param t  ::aubio_batchnorm_t layer

  \return   pointer to `fvec_t` holding the beta parameters
*/
fvec_t *aubio_batchnorm_get_beta(aubio_batchnorm_t *t);

/** get a pointer to the moving mean vector

  \param t  ::aubio_batchnorm_t layer

  \return   pointer to `fvec_t` holding the moving mean parameters

*/
fvec_t *aubio_batchnorm_get_moving_mean(aubio_batchnorm_t *t);

/** get a pointer to the moving variance vector

  \param t  ::aubio_batchnorm_t layer

  \return   pointer to `fvec_t` holding the moving variance parameters

*/
fvec_t *aubio_batchnorm_get_moving_variance(aubio_batchnorm_t *t);

/** set gamma vector of batchnorm layer

  \param t      ::aubio_batchnorm_t layer
  \param gamma  ::fvec_t containing the weights

  \return   0 on success, non-zero otherwise.

  This function will copy the content of an existing vector into
  the corresponding vector of weights in `t`.

  Note: to spare a copy and load directly the data in `t`,
  ::aubio_batchnorm_get_gamma can be used instead.

*/
uint_t aubio_batchnorm_set_gamma(aubio_batchnorm_t *t, fvec_t *gamma);

/** set beta vector of a batchnorm layer

  \param t      ::aubio_batchnorm_t layer
  \param beta   ::fvec_t containing the weights

  \return   0 on success, non-zero otherwise.

*/
uint_t aubio_batchnorm_set_beta(aubio_batchnorm_t *t, fvec_t *beta);

/** set moving mean vector of batchnorm layer

  \param t              ::aubio_batchnorm_t layer
  \param moving_mean    ::fvec_t containing the weights

  \return   0 on success, non-zero otherwise.

*/
uint_t aubio_batchnorm_set_moving_mean(aubio_batchnorm_t *t,
        fvec_t *moving_mean);

/** set moving variance vector of batchnorm layer

  \param t                  ::aubio_batchnorm_t layer
  \param moving_variance    ::fvec_t containing the weights

  \return   0 on success, non-zero otherwise.

*/
uint_t aubio_batchnorm_set_moving_variance(aubio_batchnorm_t *t,
        fvec_t *moving_variance);

/** compute batch normalization layer

  \param t             ::aubio_batchnorm_t layer
  \param input_tensor  input tensor
  \param activations   output tensor

  \return   0 on success, non-zero otherwise.

*/
void aubio_batchnorm_do(aubio_batchnorm_t *t, aubio_tensor_t *input_tensor,
        aubio_tensor_t *activations);

/** delete batch normalization layer

  \param t  ::aubio_batchnorm_t layer to delete

*/
void del_aubio_batchnorm(aubio_batchnorm_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_BATCHNORM_H */

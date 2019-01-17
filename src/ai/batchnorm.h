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

#ifdef __cplusplus
extern "C" {
#endif

/** \file

 Batch normalization layer.

 References
 ----------

 Ioffe, Sergey; Szegedy, Christian. "Batch Normalization: Accelerating Deep
 Network Training by Reducing Internal Covariate Shift", available online
 at https://arxiv.org/pdf/1502.03167.pdf

*/

typedef struct _aubio_batchnorm_t aubio_batchnorm_t;

aubio_batchnorm_t *new_aubio_batchnorm(uint_t n_outputs);

void aubio_batchnorm_do(aubio_batchnorm_t *t,
        aubio_tensor_t *input_tensor,
        aubio_tensor_t *activations);

uint_t aubio_batchnorm_set_gamma(aubio_batchnorm_t *t, fvec_t *gamma);
uint_t aubio_batchnorm_set_beta(aubio_batchnorm_t *t, fvec_t *beta);
uint_t aubio_batchnorm_set_moving_mean(aubio_batchnorm_t *t, fvec_t *moving_mean);
uint_t aubio_batchnorm_set_moving_variance(aubio_batchnorm_t *t, fvec_t *moving_variance);

fvec_t *aubio_batchnorm_get_gamma(aubio_batchnorm_t *t);
fvec_t *aubio_batchnorm_get_beta(aubio_batchnorm_t *t);
fvec_t *aubio_batchnorm_get_moving_mean(aubio_batchnorm_t *t);
fvec_t *aubio_batchnorm_get_moving_variance(aubio_batchnorm_t *t);

uint_t aubio_batchnorm_get_output_shape(aubio_batchnorm_t *t,
        aubio_tensor_t *input, uint_t *shape);

void del_aubio_batchnorm(aubio_batchnorm_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_BATCHNORM_H */

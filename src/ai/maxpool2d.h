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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_maxpool2d_t aubio_maxpool2d_t;

aubio_maxpool2d_t *new_aubio_maxpool2d(uint_t pool_size[2]);

void aubio_maxpool2d_do(aubio_maxpool2d_t *t,
        aubio_tensor_t *input_tensor,
        aubio_tensor_t *activations);

void aubio_maxpool2d_train(aubio_maxpool2d_t *t, aubio_tensor_t *input);

uint_t aubio_maxpool2d_set_weights(aubio_maxpool2d_t *t,
        aubio_tensor_t *kernels);

aubio_tensor_t *aubio_maxpool2d_get_weigths(aubio_maxpool2d_t *t);

uint_t aubio_maxpool2d_get_output_shape(aubio_maxpool2d_t *t,
        aubio_tensor_t *input, uint_t *shape);

void del_aubio_maxpool2d(aubio_maxpool2d_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_MAXPOOL2D_H */

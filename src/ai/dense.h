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

 Fully connected dense layer.

*/

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_dense_t aubio_dense_t;

aubio_dense_t *new_aubio_dense(uint_t n_units);

void del_aubio_dense(aubio_dense_t *c);

uint_t aubio_dense_get_output_shape(aubio_dense_t *c,
    aubio_tensor_t *input, uint_t *shape);

fmat_t *aubio_dense_get_weights(aubio_dense_t *c);

fvec_t *aubio_dense_get_bias(aubio_dense_t *c);

void aubio_dense_do(aubio_dense_t *c, aubio_tensor_t *input,
    aubio_tensor_t *output);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_DENSE_H */

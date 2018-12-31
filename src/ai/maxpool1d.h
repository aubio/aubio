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

#ifndef AUBIO_MAXPOOL1D_H
#define AUBIO_MAXPOOL1D_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_maxpool1d_t aubio_maxpool1d_t;

aubio_maxpool1d_t *new_aubio_maxpool1d(uint_t pool_size[1]);

uint_t aubio_maxpool1d_get_output_shape(aubio_maxpool1d_t *t,
        aubio_tensor_t *input, uint_t *shape);

void aubio_maxpool1d_do(aubio_maxpool1d_t *t,
        aubio_tensor_t *input_tensor,
        aubio_tensor_t *activations);

void del_aubio_maxpool1d(aubio_maxpool1d_t *t);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_MAXPOOL1D_H */

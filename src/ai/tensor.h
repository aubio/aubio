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

#ifndef AUBIO_TENSOR_H
#define AUBIO_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Tensor for real-valued data.

*/

#define AUBIO_TENSOR_MAXDIM 10

/** Tensor for real-valued data

  This object holds a tensor of real-valued data, ::smpl_t, with up to
  AUBIO_TENSOR_MAXDIM dimentsions.

*/
typedef struct
{
  uint_t ndim;     /**< number of dimensions */
  uint_t shape[AUBIO_TENSOR_MAXDIM]; /**< dimensions array */
  uint_t size;     /**< total number of elements */
  smpl_t *buffer;  /**< buffer of values */
  smpl_t **data;   /**< pointer to rows, or NULL when subtensor */
} aubio_tensor_t;

aubio_tensor_t *new_aubio_tensor(uint_t ndim, uint_t *shape);

void del_aubio_tensor(aubio_tensor_t *c);

uint_t aubio_tensor_as_fvec(aubio_tensor_t *c, fvec_t *o);
uint_t aubio_fvec_as_tensor(fvec_t *o, aubio_tensor_t *c);

uint_t aubio_tensor_as_fmat(aubio_tensor_t *c, fmat_t *o);
uint_t aubio_fmat_as_tensor(fmat_t *o, aubio_tensor_t *c);

uint_t aubio_tensor_get_subtensor(aubio_tensor_t *t, uint_t i,
        aubio_tensor_t *st);

smpl_t aubio_tensor_max(aubio_tensor_t *t);

#define AUBIO_ASSERT_EQUAL_SHAPE(t1, t2) { \
    AUBIO_ASSERT(t1 && t2); \
    AUBIO_ASSERT(t1->ndim == t2->ndim); \
    uint_t nn; \
    for (nn = 0; nn < t1->ndim; nn++) \
      AUBIO_ASSERT(t1->shape[nn] == t2->shape[nn]); \
    }

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_TENSOR_H */

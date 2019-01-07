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

/** create a new tensor

  \param ndim   number of dimensions
  \param shape  array of dimensions

  \return new ::aubio_tensor_t

*/
aubio_tensor_t *new_aubio_tensor(uint_t ndim, uint_t *shape);

/** destroy a tensor

  \param c  tensor to destroy

*/
void del_aubio_tensor(aubio_tensor_t *c);

/** view tensor as a vector

  \param c  tensor to view as ::fvec_t
  \param o  pointer to use to store view

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_tensor_as_fvec(aubio_tensor_t *c, fvec_t *o);

/** view vector as a tensor

  \param o  ::fvec_t to view as a tensor
  \param c  pointer to use to store view

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_fvec_as_tensor(fvec_t *o, aubio_tensor_t *c);

/** view tensor as a matrix

  \param c  tensor to view as ::fmat_t
  \param o  pointer to use to store view

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_tensor_as_fmat(aubio_tensor_t *c, fmat_t *o);

/** view matrix as a tensor

  \param o  ::fmat_t to view as a tensor
  \param c  pointer to use to store view

  \return 0 on success, non-zero otherwise

*/
uint_t aubio_fmat_as_tensor(fmat_t *o, aubio_tensor_t *c);

/** view i-th row of tensor t as a tensor

  \param t tensor to get maximum from
  \param i index of row to retrieve
  \param st subtensor to fill in

  \return 0 on success, non-zero otherwise
*/
uint_t aubio_tensor_get_subtensor(aubio_tensor_t *t, uint_t i,
        aubio_tensor_t *st);

/** find the maximum value of a tensor

  \param t tensor to get maximum from

  \return maximum value of all elements in tensor
*/
smpl_t aubio_tensor_max(aubio_tensor_t *t);

/** check if sizes of 2 tensor match

  \param t  first tensor to check size with
  \param s  second tensor to check size with

  \return 1 if tensors have the same size, 0 otherwise
*/
uint_t aubio_tensor_have_same_size(aubio_tensor_t *t, aubio_tensor_t *s);

/** print the content of a tensor

  \param t  tensor to print

 */
void aubio_tensor_print(aubio_tensor_t *t);

/** get a string representing the dimensions of this tensor

  \param t  tensor to get shape string from

  \return string of characters containing the dimensions of t
*/
const char_t *aubio_tensor_get_shape_string(aubio_tensor_t *t);

void aubio_tensor_matmul(aubio_tensor_t *a, aubio_tensor_t *b,
    aubio_tensor_t *c);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_TENSOR_H */

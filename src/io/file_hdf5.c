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

#include "aubio_priv.h"

#ifdef HAVE_HDF5

#include "fmat.h"
#include "ai/tensor.h"
#include "file_hdf5.h"

#include <hdf5.h>
#include <hdf5_hl.h>

#if !HAVE_AUBIO_DOUBLE
#define aubio_H5LTread_dataset_smpl H5LTread_dataset_float
#else
#define aubio_H5LTread_dataset_smpl H5LTread_dataset_double
#endif

struct _aubio_file_hdf5_t {
  const char_t *path;
  hid_t fid;
  hid_t datatype;
};

aubio_file_hdf5_t *new_aubio_file_hdf5(const char_t *path)
{
  aubio_file_hdf5_t *f = AUBIO_NEW(aubio_file_hdf5_t);

  f->fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f->fid <= 0) goto failure;

  f->path = path;
  return f;

failure:
  del_aubio_file_hdf5(f);
  return NULL;
}

uint_t aubio_file_hdf5_load_dataset_into_tensor (aubio_file_hdf5_t *f,
    const char_t *key, aubio_tensor_t *tensor)
{
  uint_t i;
  AUBIO_ASSERT(f && key && tensor);
  // check arguments
  if (!f->fid || !key || !tensor)
    return AUBIO_FAIL;
  // find key in file
  hid_t data_id = H5Dopen(f->fid, key, H5P_DEFAULT);
  if (data_id <= 0) {
    AUBIO_ERR("file_hdf5: failed getting key %s in %s\n", key, f->path);
    return AUBIO_FAIL;
  }
  // get dimensions
  hsize_t shape[10];
  hid_t space = H5Dget_space(data_id);
  int ndim = H5Sget_simple_extent_dims(space, shape, NULL);
  if (ndim <= 0) {
    AUBIO_ERR("file_hdf5: failed to get dims of %s in %s\n", key, f->path);
    return AUBIO_FAIL;
  }

  // check output tensor dimension matches
  AUBIO_ASSERT(ndim == (sint_t)tensor->ndim);
  for (i = 0; i < (uint_t)ndim; i++) {
    AUBIO_ASSERT(shape[i] == tensor->shape[i]);
  }

  if (ndim != (sint_t)tensor->ndim) return AUBIO_FAIL;
  for (i = 0; i < (uint_t)ndim; i++) {
    if (shape[i] != tensor->shape[i]) return AUBIO_FAIL;
  }

  // read data from hdf5 file into tensor buffer
  smpl_t *buffer = tensor->buffer;
  herr_t err = aubio_H5LTread_dataset_smpl(f->fid, key, buffer);

  if (err < 0) {
    return AUBIO_FAIL;
  }

  //AUBIO_DBG("file_hdf5: loaded : shape %s from key %s\n",
  //    aubio_tensor_get_shape_string(tensor), key);

  H5Dclose(data_id);
  return AUBIO_OK;
}

uint_t aubio_file_hdf5_load_dataset_into_matrix(aubio_file_hdf5_t *f,
    const char_t *key, fmat_t *mat) {
  aubio_tensor_t t;
  if (aubio_fmat_as_tensor (mat, &t)) return AUBIO_FAIL;
  return aubio_file_hdf5_load_dataset_into_tensor(f, key, &t);
}


uint_t aubio_file_hdf5_load_dataset_into_vector(aubio_file_hdf5_t *f,
    const char_t *key, fvec_t *vec) {
  aubio_tensor_t t;
  if (aubio_fvec_as_tensor (vec, &t)) return AUBIO_FAIL;
  return aubio_file_hdf5_load_dataset_into_tensor(f, key, &t);
}

void del_aubio_file_hdf5(aubio_file_hdf5_t *f)
{
  AUBIO_ASSERT(f);
  if (f->fid > 0)
    H5Fclose(f->fid);
  AUBIO_FREE(f);
}

#endif /* HAVE_HDF5 */

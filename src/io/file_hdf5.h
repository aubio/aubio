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

#ifndef AUBIO_FILE_HDF5_H
#define AUBIO_FILE_HDF5_H

typedef struct _aubio_file_hdf5_t aubio_file_hdf5_t;

aubio_file_hdf5_t *new_aubio_file_hdf5(const char_t *path);

void del_aubio_file_hdf5(aubio_file_hdf5_t *f);

uint_t aubio_file_hdf5_load_dataset_into_tensor (aubio_file_hdf5_t *f,
    const char_t *key, aubio_tensor_t *tensor);

uint_t aubio_file_hdf5_load_dataset_into_matrix(aubio_file_hdf5_t *f,
    const char_t *key, fmat_t *mat);

uint_t aubio_file_hdf5_load_dataset_into_vector (aubio_file_hdf5_t *f,
    const char_t *key, fvec_t *vec);

#endif /* AUBIO_FILE_HDF5_H */

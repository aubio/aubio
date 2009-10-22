/*
  Copyright (C) 2003-2009 Paul Brossier <piem@aubio.org>

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

#ifndef _FVEC_H
#define _FVEC_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Real buffers

  This file specifies the fvec_t buffer type, which is used throughout aubio to
  store real data.

*/

/** Buffer for real data */
typedef struct {
  uint_t length;   /**< length of buffer */
  uint_t channels; /**< number of channels */
  smpl_t **data;   /**< data array of size [length] * [channels] */
} fvec_t;

/** fvec_t buffer creation function

  \param length the length of the buffer to create
  \param channels the number of channels in the buffer

*/
fvec_t * new_fvec(uint_t length, uint_t channels);
/** fvec_t buffer deletion function

  \param s buffer to delete as returned by new_fvec()

*/
void del_fvec(fvec_t *s);
/** read sample value in a buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained using vec->data[channel][position]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to read from
  \param channel channel to read from
  \param position sample position to read from 

*/
smpl_t fvec_read_sample(fvec_t *s, uint_t channel, uint_t position);
/** write sample value in a buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->data[channel][position]. Its purpose
  is to access these values from wrappers, as created by swig.

  \param s vector to write to 
  \param data value to write in s->data[channel][position]
  \param channel channel to write to 
  \param position sample position to write to 

*/
void  fvec_write_sample(fvec_t *s, smpl_t data, uint_t channel, uint_t position);
/** read channel vector from a buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->data[channel]. Its purpose is to access
  these values from wrappers, as created by swig.

  \param s vector to read from
  \param channel channel to read from

*/
smpl_t * fvec_get_channel(fvec_t *s, uint_t channel);
/** write channel vector into a buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->data[channel]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to write to 
  \param data vector of [length] values to write
  \param channel channel to write to 

*/
void fvec_put_channel(fvec_t *s, smpl_t * data, uint_t channel);
/** read data from a buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->data. Its purpose is to access these values
  from wrappers, as created by swig.

  \param s vector to read from

*/
smpl_t ** fvec_get_data(fvec_t *s);

/** print out fvec data 

  \param s vector to print out 

*/
void fvec_print(fvec_t *s);

/** set all elements to a given value

  \param s vector to modify
  \param val value to set elements to

*/
void fvec_set(fvec_t *s, smpl_t val);

/** set all elements to zero 

  \param s vector to modify

*/
void fvec_zeros(fvec_t *s);

/** set all elements to ones 

  \param s vector to modify

*/
void fvec_ones(fvec_t *s);

/** revert order of vector elements

  \param s vector to revert

*/
void fvec_rev(fvec_t *s);

/** apply weight to vector

  If the weight vector is longer than s, only the first elements are used. If
  the weight vector is shorter than s, the last elements of s are not weighted.

  \param s vector to weight
  \param weight weighting coefficients

*/
void fvec_weight(fvec_t *s, fvec_t *weight);

/** make a copy of a vector

  \param s source vector
  \param t vector to copy to

*/
void fvec_copy(fvec_t *s, fvec_t *t);

#ifdef __cplusplus
}
#endif

#endif /* _FVEC_H */

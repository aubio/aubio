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

#ifndef _CVEC_H
#define _CVEC_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Vector of complex-valued data

  This file specifies the ::cvec_t buffer type, which is used throughout aubio
  to store complex data. Complex values are stored in terms of ::cvec_t.phas
  and norm, within size/2+1 long vectors of ::smpl_t.

  \example test-cvec.c

*/

/** Buffer for complex data

  \code

  uint_t buffer_size = 1024;

  // create a complex vector of 512 values
  cvec_t * input = new_cvec (buffer_size);

  // set some values of the vector
  input->norm[23] = 2.;
  input->phas[23] = M_PI;
  // ..

  // compute the mean of the vector
  mean = cvec_mean(input);

  // destroy the vector
  del_cvec (input);

  \endcode

 */
typedef struct {
  uint_t length;  /**< length of buffer = (requested length)/2 + 1 */
  smpl_t *norm;   /**< norm array of size ::cvec_t.length */
  smpl_t *phas;   /**< phase array of size ::cvec_t.length */
} cvec_t;

/** cvec_t buffer creation function

  This function creates a cvec_t structure holding two arrays of size
  [length/2+1], corresponding to the norm and phase values of the
  spectral frame. The length stored in the structure is the actual size of both
  arrays, not the length of the complex and symmetrical vector, specified as
  creation argument.

  \param length the length of the buffer to create

*/
cvec_t * new_cvec(uint_t length);
/** cvec_t buffer deletion function

  \param s buffer to delete as returned by new_cvec()

*/
void del_cvec(cvec_t *s);
/** write norm value in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->norm[position]. Its purpose
  is to access these values from wrappers, as created by swig.

  \param s vector to write to
  \param data norm value to write in s->norm[position]
  \param position sample position to write to

*/
void cvec_write_norm(cvec_t *s, smpl_t data, uint_t position);
/** write phase value in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->phas[position]. Its purpose
  is to access these values from wrappers, as created by swig.

  \param s vector to write to
  \param data phase value to write in s->phas[position]
  \param position sample position to write to

*/
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t position);
/** read norm value from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->norm[position]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to read from
  \param position sample position to read from

*/
smpl_t cvec_read_norm(cvec_t *s, uint_t position);
/** read phase value from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->phas[position]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to read from
  \param position sample position to read from

*/
smpl_t cvec_read_phas(cvec_t *s, uint_t position);
/** read norm data from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->norm. Its purpose is to access these values
  from wrappers, as created by swig.

  \param s vector to read from

*/
smpl_t * cvec_get_norm(cvec_t *s);
/** read phase data from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->phas. Its purpose is to access these values
  from wrappers, as created by swig.

  \param s vector to read from

*/
smpl_t * cvec_get_phas(cvec_t *s);

/** print out cvec data

  \param s vector to print out

*/
void cvec_print(cvec_t *s);

/** set all elements to a given value

  \param s vector to modify
  \param val value to set elements to

*/
void cvec_set(cvec_t *s, smpl_t val);

/** set all elements to zero

  \param s vector to modify

*/
void cvec_zeros(cvec_t *s);

/** set all elements to ones

  \param s vector to modify

*/
void cvec_ones(cvec_t *s);

#ifdef __cplusplus
}
#endif

#endif /* _CVEC_H */


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
#define _CVEC_H_

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Complex buffers

  This file specifies the cvec_t buffer type, which is used throughout aubio to
  store complex data. Complex values are stored in terms of phase and
  norm, within size/2+1 long vectors.

*/

/** Buffer for complex data */
typedef struct {
  uint_t length;   /**< length of buffer = (requested length)/2 + 1 */
  uint_t channels; /**< number of channels */
  smpl_t **norm;   /**< norm array of size [length] * [channels] */
  smpl_t **phas;   /**< phase array of size [length] * [channels] */
} cvec_t;

/** cvec_t buffer creation function

  This function creates a cvec_t structure holding two arrays of size
  [length/2+1] * channels, corresponding to the norm and phase values of the
  spectral frame. The length stored in the structure is the actual size of both
  arrays, not the length of the complex and symetrical vector, specified as
  creation argument.

  \param length the length of the buffer to create
  \param channels the number of channels in the buffer

*/
cvec_t * new_cvec(uint_t length, uint_t channels);
/** cvec_t buffer deletion function

  \param s buffer to delete as returned by new_cvec()

*/
void del_cvec(cvec_t *s);
/** write norm value in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->norm[channel][position]. Its purpose
  is to access these values from wrappers, as created by swig.

  \param s vector to write to 
  \param data norm value to write in s->norm[channel][position]
  \param channel channel to write to 
  \param position sample position to write to

*/
void cvec_write_norm(cvec_t *s, smpl_t data, uint_t channel, uint_t position);
/** write phase value in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->phas[channel][position]. Its purpose
  is to access these values from wrappers, as created by swig.

  \param s vector to write to
  \param data phase value to write in s->phas[channel][position]
  \param channel channel to write to
  \param position sample position to write to

*/
void cvec_write_phas(cvec_t *s, smpl_t data, uint_t channel, uint_t position);
/** read norm value from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->norm[channel][position]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to read from
  \param channel channel to read from
  \param position sample position to read from

*/
smpl_t cvec_read_norm(cvec_t *s, uint_t channel, uint_t position);
/** read phase value from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->phas[channel][position]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to read from
  \param channel channel to read from
  \param position sample position to read from

*/
smpl_t cvec_read_phas(cvec_t *s, uint_t channel, uint_t position);
/** write norm channel in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->norm[channel]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to write to
  \param data norm vector of [length] samples to write in s->norm[channel]
  \param channel channel to write to

*/
void cvec_put_norm_channel(cvec_t *s, smpl_t * data, uint_t channel);
/** write phase channel in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained by assigning vec->phas[channel]. Its purpose is to
  access these values from wrappers, as created by swig.

  \param s vector to write to
  \param data phase vector of [length] samples to write in s->phas[channel]
  \param channel channel to write to

*/
void cvec_put_phas_channel(cvec_t *s, smpl_t * data, uint_t channel);
/** read norm channel from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->norm[channel]. Its purpose is to access
  these values from wrappers, as created by swig.

  \param s vector to read from 
  \param channel channel to read from

*/
smpl_t * cvec_get_norm_channel(cvec_t *s, uint_t channel);
/** write phase channel in a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->phas[channel]. Its purpose is to access
  these values from wrappers, as created by swig.

  \param s vector to read from 
  \param channel channel to read from 

*/
smpl_t * cvec_get_phas_channel(cvec_t *s, uint_t channel);
/** read norm data from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->norm. Its purpose is to access these values
  from wrappers, as created by swig.

  \param s vector to read from

*/
smpl_t ** cvec_get_norm(cvec_t *s);
/** read phase data from a complex buffer

  Note that this function is not used in the aubio library, since the same
  result can be obtained with vec->phas. Its purpose is to access these values
  from wrappers, as created by swig.

  \param s vector to read from

*/
smpl_t ** cvec_get_phas(cvec_t *s);

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


/*
   Copyright (C) 2003 Paul Brossier

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

*/

#ifndef _SAMPLE_H
#define _SAMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Real and complex buffers

  This file specifies fvec_t and cvec_t buffers types, which are used
  throughout aubio to store real and complex data. Complex values are stored in
  terms of phase and norm.

*/

/** Sample buffer type */
typedef struct _fvec_t fvec_t;
/** Spectrum buffer type */
typedef struct _cvec_t cvec_t;
/** Buffer for real values */
struct _fvec_t {
  uint_t length;   /**< length of buffer */
  uint_t channels; /**< number of channels */
  smpl_t **data;   /**< data array of size [length] * [channels] */
};
/** Buffer for complex data */
struct _cvec_t {
  uint_t length;   /**< length of buffer = (requested length)/2 + 1 */
  uint_t channels; /**< number of channels */
  smpl_t **norm;   /**< norm array of size [length] * [channels] */
  smpl_t **phas;   /**< phase array of size [length] * [channels] */
};
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

#ifdef __cplusplus
}
#endif

#endif /* _SAMPLE_H */

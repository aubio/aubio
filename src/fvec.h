/*
   Copyright (C) 2003-2007 Paul Brossier <piem@piem.org>

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

/** Sample buffer type */
typedef struct _fvec_t fvec_t;
/** Buffer for real values */
struct _fvec_t {
  uint_t length;   /**< length of buffer */
  uint_t channels; /**< number of channels */
  smpl_t **data;   /**< data array of size [length] * [channels] */
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

#ifdef __cplusplus
}
#endif

#endif /* _FVEC_H */

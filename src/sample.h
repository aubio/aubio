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

/** @file
 * Sample buffers
 *
 * Create fvec_t and cvec_t buffers 
 */

/**
 * Sample buffer type
 * 	 
 *   contains length and channels number
 */
typedef struct _fvec_t fvec_t;
/** 
 * Spectrum buffer type
 *
 *   contains length and channels number
 */ 
typedef struct _cvec_t cvec_t;

/**
 * Buffer for audio samples
 */
struct _fvec_t {
  uint_t length;
  uint_t channels;
  smpl_t **data;
};

/**
 * Buffer for spectral data
 */
struct _cvec_t {
  uint_t length;
  uint_t channels;
  smpl_t **norm;
  smpl_t **phas;
};


/* buffer function */
extern fvec_t * new_fvec(uint_t length, uint_t channels);
extern void del_fvec(fvec_t *s);
smpl_t fvec_read_sample(fvec_t *s, uint_t channel, uint_t position);
void  fvec_write_sample(fvec_t *s, smpl_t data, uint_t channel, uint_t position);
smpl_t * fvec_get_channel(fvec_t *s, uint_t channel);
smpl_t ** fvec_get_data(fvec_t *s);
void fvec_put_channel(fvec_t *s, smpl_t * data, uint_t channel);
extern cvec_t * new_cvec(uint_t length, uint_t channels);
extern void del_cvec(cvec_t *s);

#ifdef __cplusplus
}
#endif

#endif /* _SAMPLE_H */

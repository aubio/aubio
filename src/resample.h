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

#ifndef _RESAMPLE_H
#define _RESAMPLE_H

/** \file
 
 Resampling object

 This object resamples an input vector into an output vector using
 libsamplerate. See http://www.mega-nerd.com/SRC/
 
*/

#ifdef __cplusplus
extern "C" {
#endif

/** resampler object */
typedef struct _aubio_resampler_t aubio_resampler_t;
/** create resampler object 

  \param ratio output_sample_rate / input_sample_rate 
  \param type libsamplerate resampling type

*/
aubio_resampler_t * new_aubio_resampler(float ratio, uint_t type);
/** delete resampler object */
void del_aubio_resampler(aubio_resampler_t *s);
/** resample input in output

  \param s resampler object
  \param input input buffer of size N
  \param output output buffer of size N*ratio

*/
uint_t aubio_resampler_process(aubio_resampler_t *s, fvec_t * input,  fvec_t * output);

#ifdef __cplusplus
}
#endif

#endif /* _RESAMPLE_H */

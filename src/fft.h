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

/** @file 
 * Fft object (fftw3f)
 * */

#ifndef FFT_H_
#define FFT_H_
/*
// complex before fftw3 
#include <complex.h>
*/
#include <fftw3.h>

#if FFTW3F_SUPPORT
#define FFTW_TYPE fftwf_complex
#else
#define FFTW_TYPE fftw_complex
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef FFTW_TYPE fft_data_t;

typedef struct _aubio_fft_t aubio_fft_t;

/* fftw funcs */
extern aubio_fft_t * new_aubio_fft(uint_t size);
extern void del_aubio_fft(aubio_fft_t * s);
extern void aubio_fft_do (const aubio_fft_t *s, const smpl_t * data,
    fft_data_t * spectrum, const uint_t size);
extern void aubio_fft_rdo(const aubio_fft_t *s, const fft_data_t * spectrum,
    smpl_t * data, const uint_t size);
/** get norm from spectrum */
void aubio_fft_getnorm(smpl_t * norm, fft_data_t * spectrum, uint_t size);
/** get phase from spectrum */
void aubio_fft_getphas(smpl_t * phase, fft_data_t * spectrum, uint_t size);

#ifdef __cplusplus
}
#endif

#endif

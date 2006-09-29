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

/** \file 

  Fast Fourier Transform object

*/

#ifndef FFT_H_
#define FFT_H_

/* note that <complex.h> is not included here but only in aubio_priv.h, so that
 * c++ projects can still use their own complex definition. */
#include <fftw3.h>

#ifdef HAVE_COMPLEX_H
#if FFTW3F_SUPPORT
#define FFTW_TYPE fftwf_complex
#else
#define FFTW_TYPE fftw_complex
#endif
#else
#if FFTW3F_SUPPORT
/** fft data type */
#define FFTW_TYPE float
#else
/** fft data type */
#define FFTW_TYPE double
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** fft data type */
typedef FFTW_TYPE fft_data_t;

/** FFT object
 
  This object computes forward and backward FFTs, using the complex type to
  store the results. The phase vocoder or aubio_mfft_t objects should be
  preferred to using directly aubio_fft_t. The FFT are computed using FFTW3
  (although support for another library could be added).

*/
typedef struct _aubio_fft_t aubio_fft_t;

/** create new FFT computation object

  \param size length of the FFT

*/
aubio_fft_t * new_aubio_fft(uint_t size);
/** delete FFT object 

  \param s fft object as returned by new_aubio_fft

*/
void del_aubio_fft(aubio_fft_t * s);
/** compute forward FFT

  \param s fft object as returned by new_aubio_fft
  \param data input signal 
  \param spectrum output spectrum 
  \param size length of the input vector 

*/
void aubio_fft_do (const aubio_fft_t *s, const smpl_t * data,
    fft_data_t * spectrum, const uint_t size);
/** compute backward (inverse) FFT

  \param s fft object as returned by new_aubio_fft
  \param spectrum input spectrum 
  \param data output signal 
  \param size length of the input vector 

*/
void aubio_fft_rdo(const aubio_fft_t *s, const fft_data_t * spectrum,
    smpl_t * data, const uint_t size);
/** compute norm vector from input spectrum

  \param norm magnitude vector output
  \param spectrum spectral data input
  \param size size of the vectors

*/
void aubio_fft_getnorm(smpl_t * norm, fft_data_t * spectrum, uint_t size);
/** compute phase vector from input spectrum 
 
  \param phase phase vector output
  \param spectrum spectral data input
  \param size size of the vectors

*/
void aubio_fft_getphas(smpl_t * phase, fft_data_t * spectrum, uint_t size);

/** FFT object (using cvec)

  This object works similarly as aubio_fft_t, except the spectral data is
  stored in a cvec_t as two vectors, magnitude and phase. 

*/
typedef struct _aubio_mfft_t aubio_mfft_t;

/** create new FFT computation object

  \param winsize length of the FFT
  \param channels number of channels 

*/
aubio_mfft_t * new_aubio_mfft(uint_t winsize, uint_t channels);
/** compute forward FFT

  \param fft fft object as returned by new_aubio_mfft
  \param in input signal 
  \param fftgrain output spectrum

*/
void aubio_mfft_do (aubio_mfft_t * fft,fvec_t * in,cvec_t * fftgrain);
/** compute backward (inverse) FFT

  \param fft fft object as returned by new_aubio_mfft
  \param fftgrain input spectrum (cvec) 
  \param out output signal 

*/
void aubio_mfft_rdo(aubio_mfft_t * fft,cvec_t * fftgrain, fvec_t * out);
/** delete FFT object 

  \param fft fft object as returned by new_aubio_mfft

*/
void del_aubio_mfft(aubio_mfft_t * fft);


#ifdef __cplusplus
}
#endif

#endif

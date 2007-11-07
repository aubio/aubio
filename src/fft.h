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
  \param channels number of channels

*/
aubio_fft_t * new_aubio_fft(uint_t size, uint_t channels);
/** delete FFT object 

  \param s fft object as returned by new_aubio_fft

*/
void del_aubio_fft(aubio_fft_t * s);

/** compute forward FFT

  \param s fft object as returned by new_aubio_fft
  \param input input signal 
  \param spectrum output spectrum 

*/
void aubio_fft_do (aubio_fft_t *s, fvec_t * input, cvec_t * spectrum);
/** compute backward (inverse) FFT

  \param s fft object as returned by new_aubio_fft
  \param spectrum input spectrum 
  \param output output signal 

*/
void aubio_fft_rdo (aubio_fft_t *s, cvec_t * spectrum, fvec_t * output);

/** compute forward FFT

  \param s fft object as returned by new_aubio_fft
  \param input real input signal 
  \param compspec complex output fft real/imag

*/
void aubio_fft_do_complex (aubio_fft_t *s, fvec_t * input, fvec_t * compspec);
/** compute backward (inverse) FFT from real/imag

  \param s fft object as returned by new_aubio_fft
  \param compspec real/imag input fft array 
  \param output real output array 

*/
void aubio_fft_rdo_complex (aubio_fft_t *s, fvec_t * compspec, fvec_t * output);

/** convert real/imag spectrum to norm/phas spectrum 

  \param compspec real/imag input fft array 
  \param spectrum cvec norm/phas output array 

*/
void aubio_fft_get_spectrum(fvec_t * compspec, cvec_t * spectrum);
/** convert real/imag spectrum to norm/phas spectrum 

  \param compspec real/imag input fft array 
  \param spectrum cvec norm/phas output array 

*/
void aubio_fft_get_realimag(cvec_t * spectrum, fvec_t * compspec);

/** compute phas spectrum from real/imag parts 

  \param compspec real/imag input fft array 
  \param spectrum cvec norm/phas output array 

*/
void aubio_fft_get_phas(fvec_t * compspec, cvec_t * spectrum);
/** compute imaginary part from the norm/phas cvec 

  \param spectrum norm/phas input array 
  \param compspec real/imag output fft array 

*/
void aubio_fft_get_imag(cvec_t * spectrum, fvec_t * compspec);

/** compute norm component from real/imag parts 

  \param compspec real/imag input fft array 
  \param spectrum cvec norm/phas output array 

*/
void aubio_fft_get_norm(fvec_t * compspec, cvec_t * spectrum);
/** compute real part from norm/phas components 

  \param spectrum norm/phas input array 
  \param compspec real/imag output fft array 

*/
void aubio_fft_get_real(cvec_t * spectrum, fvec_t * compspec);

#ifdef __cplusplus
}
#endif

#endif // FFT_H_

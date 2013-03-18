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

/** \file 

  Fast Fourier Transform

  \example src/spectral/test-fft.c

*/

#ifndef FFT_H_
#define FFT_H_

#ifdef __cplusplus
extern "C" {
#endif

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
aubio_fft_t * new_aubio_fft (uint_t size);
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

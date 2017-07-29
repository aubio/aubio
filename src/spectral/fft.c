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

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "mathutils.h"
#include "spectral/fft.h"

#ifdef HAVE_FFTW3             // using FFTW3
/* note that <complex.h> is not included here but only in aubio_priv.h, so that
 * c++ projects can still use their own complex definition. */
#include <fftw3.h>
#include <pthread.h>

#ifdef HAVE_COMPLEX_H
#ifdef HAVE_FFTW3F
/** fft data type with complex.h and fftw3f */
#define FFTW_TYPE fftwf_complex
#else
/** fft data type with complex.h and fftw3 */
#define FFTW_TYPE fftw_complex
#endif
#else
#ifdef HAVE_FFTW3F
/** fft data type without complex.h and with fftw3f */
#define FFTW_TYPE float
#else
/** fft data type without complex.h and with fftw */
#define FFTW_TYPE double
#endif
#endif

/** fft data type */
typedef FFTW_TYPE fft_data_t;

#ifdef HAVE_FFTW3F
#define fftw_malloc            fftwf_malloc
#define fftw_free              fftwf_free
#define fftw_execute           fftwf_execute
#define fftw_plan_dft_r2c_1d   fftwf_plan_dft_r2c_1d
#define fftw_plan_dft_c2r_1d   fftwf_plan_dft_c2r_1d
#define fftw_plan_r2r_1d       fftwf_plan_r2r_1d
#define fftw_plan              fftwf_plan
#define fftw_destroy_plan      fftwf_destroy_plan
#endif

#ifdef HAVE_FFTW3F
#if HAVE_AUBIO_DOUBLE
#error "Using aubio in double precision with fftw3 in single precision"
#endif /* HAVE_AUBIO_DOUBLE */
#define real_t float
#elif defined (HAVE_FFTW3) /* HAVE_FFTW3F */
#if !HAVE_AUBIO_DOUBLE
#error "Using aubio in single precision with fftw3 in double precision"
#endif /* HAVE_AUBIO_DOUBLE */
#define real_t double
#endif /* HAVE_FFTW3F */

// a global mutex for FFTW thread safety
pthread_mutex_t aubio_fftw_mutex = PTHREAD_MUTEX_INITIALIZER;

#elif defined HAVE_ACCELERATE        // using ACCELERATE
// https://developer.apple.com/library/mac/#documentation/Accelerate/Reference/vDSPRef/Reference/reference.html
#include <Accelerate/Accelerate.h>

#if !HAVE_AUBIO_DOUBLE
#define aubio_vDSP_ctoz                vDSP_ctoz
#define aubio_vDSP_fft_zrip            vDSP_fft_zrip
#define aubio_vDSP_ztoc                vDSP_ztoc
#define aubio_vDSP_zvmags              vDSP_zvmags
#define aubio_vDSP_zvphas              vDSP_zvphas
#define aubio_vDSP_vsadd               vDSP_vsadd
#define aubio_vDSP_vsmul               vDSP_vsmul
#define aubio_vDSP_create_fftsetup     vDSP_create_fftsetup
#define aubio_vDSP_destroy_fftsetup    vDSP_destroy_fftsetup
#define aubio_DSPComplex               DSPComplex
#define aubio_DSPSplitComplex          DSPSplitComplex
#define aubio_FFTSetup                 FFTSetup
#define aubio_vvsqrt                   vvsqrtf
#else
#define aubio_vDSP_ctoz                vDSP_ctozD
#define aubio_vDSP_fft_zrip            vDSP_fft_zripD
#define aubio_vDSP_ztoc                vDSP_ztocD
#define aubio_vDSP_zvmags              vDSP_zvmagsD
#define aubio_vDSP_zvphas              vDSP_zvphasD
#define aubio_vDSP_vsadd               vDSP_vsaddD
#define aubio_vDSP_vsmul               vDSP_vsmulD
#define aubio_vDSP_create_fftsetup     vDSP_create_fftsetupD
#define aubio_vDSP_destroy_fftsetup    vDSP_destroy_fftsetupD
#define aubio_DSPComplex               DSPDoubleComplex
#define aubio_DSPSplitComplex          DSPDoubleSplitComplex
#define aubio_FFTSetup                 FFTSetupD
#define aubio_vvsqrt                   vvsqrt
#endif /* HAVE_AUBIO_DOUBLE */

#elif defined HAVE_INTEL_IPP // using INTEL IPP

#include <ippcore.h>
#include <ippvm.h>
#include <ipps.h>

#else // using OOURA
// let's use ooura instead
extern void aubio_ooura_rdft(int, int, smpl_t *, int *, smpl_t *);

#endif

struct _aubio_fft_t {
  uint_t winsize;
  uint_t fft_size;

#ifdef HAVE_FFTW3             // using FFTW3
  real_t *in, *out;
  fftw_plan pfw, pbw;
  fft_data_t * specdata; /* complex spectral data */

#elif defined HAVE_ACCELERATE  // using ACCELERATE
  int log2fftsize;
  aubio_FFTSetup fftSetup;
  aubio_DSPSplitComplex spec;
  smpl_t *in, *out;
  
#elif defined HAVE_INTEL_IPP  // using Intel IPP
  // mark FFT impl as Intel IPP
  #define INTEL_IPP_FFT 1
  smpl_t *in, *out;
  Ipp8u* memSpec;
  Ipp8u* memInit;
  Ipp8u* memBuffer;
  #if HAVE_AUBIO_DOUBLE
    struct FFTSpec_R_64f* fftSpec;
    Ipp64fc* complexOut;
  #else
    struct FFTSpec_R_32f* fftSpec;
    Ipp32fc* complexOut;
  #endif
#else                         // using OOURA
  smpl_t *in, *out;
  smpl_t *w;
  int *ip;
#endif /* using OOURA */

  fvec_t * compspec;
};

aubio_fft_t * new_aubio_fft (uint_t winsize) {
  aubio_fft_t * s = AUBIO_NEW(aubio_fft_t);
  if ((sint_t)winsize < 2) {
    AUBIO_ERR("fft: got winsize %d, but can not be < 2\n", winsize);
    goto beach;
  }

#ifdef HAVE_FFTW3
  uint_t i;
  s->winsize  = winsize;
  /* allocate memory */
  s->in       = AUBIO_ARRAY(real_t,winsize);
  s->out      = AUBIO_ARRAY(real_t,winsize);
  s->compspec = new_fvec(winsize);
  /* create plans */
  pthread_mutex_lock(&aubio_fftw_mutex);
#ifdef HAVE_COMPLEX_H
  s->fft_size = winsize/2 + 1;
  s->specdata = (fft_data_t*)fftw_malloc(sizeof(fft_data_t)*s->fft_size);
  s->pfw = fftw_plan_dft_r2c_1d(winsize, s->in,  s->specdata, FFTW_ESTIMATE);
  s->pbw = fftw_plan_dft_c2r_1d(winsize, s->specdata, s->out, FFTW_ESTIMATE);
#else
  s->fft_size = winsize;
  s->specdata = (fft_data_t*)fftw_malloc(sizeof(fft_data_t)*s->fft_size);
  s->pfw = fftw_plan_r2r_1d(winsize, s->in,  s->specdata, FFTW_R2HC, FFTW_ESTIMATE);
  s->pbw = fftw_plan_r2r_1d(winsize, s->specdata, s->out, FFTW_HC2R, FFTW_ESTIMATE);
#endif
  pthread_mutex_unlock(&aubio_fftw_mutex);
  for (i = 0; i < s->winsize; i++) {
    s->in[i] = 0.;
    s->out[i] = 0.;
  }
  for (i = 0; i < s->fft_size; i++) {
    s->specdata[i] = 0.;
  }

#elif defined HAVE_ACCELERATE  // using ACCELERATE
  s->winsize = winsize;
  s->fft_size = winsize;
  s->compspec = new_fvec(winsize);
  s->log2fftsize = aubio_power_of_two_order(s->fft_size);
  s->in = AUBIO_ARRAY(smpl_t, s->fft_size);
  s->out = AUBIO_ARRAY(smpl_t, s->fft_size);
  s->spec.realp = AUBIO_ARRAY(smpl_t, s->fft_size/2);
  s->spec.imagp = AUBIO_ARRAY(smpl_t, s->fft_size/2);
  s->fftSetup = aubio_vDSP_create_fftsetup(s->log2fftsize, FFT_RADIX2);

#elif defined HAVE_INTEL_IPP  // using Intel IPP
  const IppHintAlgorithm qualityHint = ippAlgHintAccurate; // OR ippAlgHintFast;
  const int flags = IPP_FFT_NODIV_BY_ANY; // we're scaling manually afterwards 
  int order = aubio_power_of_two_order(winsize);
  int sizeSpec, sizeInit, sizeBuffer;
  IppStatus status;

  if (winsize <= 4 || aubio_is_power_of_two(winsize) != 1)
  {
    AUBIO_ERR("intel IPP fft: can only create with sizes > 4 and power of two, requested %d,"
      " try recompiling aubio with --enable-fftw3\n", winsize);
    goto beach;
  }

#if HAVE_AUBIO_DOUBLE
  status = ippsFFTGetSize_R_64f(order, flags, qualityHint,
      &sizeSpec, &sizeInit, &sizeBuffer);
#else
  status = ippsFFTGetSize_R_32f(order, flags, qualityHint,
    &sizeSpec, &sizeInit, &sizeBuffer);
#endif
  if (status != ippStsNoErr) {
    AUBIO_ERR("fft: failed to initialize fft. IPP error: %d\n", status);
    goto beach;
  }
  s->fft_size = s->winsize = winsize;
  s->compspec = new_fvec(winsize);
  s->in = AUBIO_ARRAY(smpl_t, s->winsize);
  s->out = AUBIO_ARRAY(smpl_t, s->winsize);
  s->memSpec = ippsMalloc_8u(sizeSpec);
  s->memBuffer = ippsMalloc_8u(sizeBuffer);
  if (sizeInit > 0 ) {
    s->memInit = ippsMalloc_8u(sizeInit);
  }
#if HAVE_AUBIO_DOUBLE
  s->complexOut = ippsMalloc_64fc(s->fft_size / 2 + 1);
  status = ippsFFTInit_R_64f(
    &s->fftSpec, order, flags, qualityHint, s->memSpec, s->memInit);
#else
  s->complexOut = ippsMalloc_32fc(s->fft_size / 2 + 1);
  status = ippsFFTInit_R_32f(
    &s->fftSpec, order, flags, qualityHint, s->memSpec, s->memInit);
#endif
  if (status != ippStsNoErr) {
    AUBIO_ERR("fft: failed to initialize. IPP error: %d\n", status);
    goto beach;
  }

#else                         // using OOURA
  if (aubio_is_power_of_two(winsize) != 1) {
    AUBIO_ERR("fft: can only create with sizes power of two, requested %d,"
        " try recompiling aubio with --enable-fftw3\n", winsize);
    goto beach;
  }
  s->winsize = winsize;
  s->fft_size = winsize / 2 + 1;
  s->compspec = new_fvec(winsize);
  s->in    = AUBIO_ARRAY(smpl_t, s->winsize);
  s->out   = AUBIO_ARRAY(smpl_t, s->winsize);
  s->ip    = AUBIO_ARRAY(int   , s->fft_size);
  s->w     = AUBIO_ARRAY(smpl_t, s->fft_size);
  s->ip[0] = 0;
#endif /* using OOURA */

  return s;

beach:
  AUBIO_FREE(s);
  return NULL;
}

void del_aubio_fft(aubio_fft_t * s) {
  /* destroy data */
#ifdef HAVE_FFTW3             // using FFTW3
  pthread_mutex_lock(&aubio_fftw_mutex);
  fftw_destroy_plan(s->pfw);
  fftw_destroy_plan(s->pbw);
  fftw_free(s->specdata);
  pthread_mutex_unlock(&aubio_fftw_mutex);

#elif defined HAVE_ACCELERATE // using ACCELERATE
  AUBIO_FREE(s->spec.realp);
  AUBIO_FREE(s->spec.imagp);
  aubio_vDSP_destroy_fftsetup(s->fftSetup);

#elif defined HAVE_INTEL_IPP  // using Intel IPP
  ippFree(s->memSpec);
  ippFree(s->memInit);
  ippFree(s->memBuffer);
  ippFree(s->complexOut);

#else                         // using OOURA
  AUBIO_FREE(s->w);
  AUBIO_FREE(s->ip);
#endif

  del_fvec(s->compspec);
  AUBIO_FREE(s->in);
  AUBIO_FREE(s->out);
  AUBIO_FREE(s);
}

void aubio_fft_do(aubio_fft_t * s, const fvec_t * input, cvec_t * spectrum) {
  aubio_fft_do_complex(s, input, s->compspec);
  aubio_fft_get_spectrum(s, s->compspec, spectrum);
}

void aubio_fft_rdo(aubio_fft_t * s, const cvec_t * spectrum, fvec_t * output) {
  aubio_fft_get_realimag(s, spectrum, s->compspec);
  aubio_fft_rdo_complex(s, s->compspec, output);
}

void aubio_fft_do_complex(aubio_fft_t * s, const fvec_t * input, fvec_t * compspec) {
  uint_t i;
#ifndef HAVE_MEMCPY_HACKS
  for (i=0; i < s->winsize; i++) {
    s->in[i] = input->data[i];
  }
#else
  memcpy(s->in, input->data, s->winsize * sizeof(smpl_t));
#endif /* HAVE_MEMCPY_HACKS */

#ifdef HAVE_FFTW3             // using FFTW3
  fftw_execute(s->pfw);
#ifdef HAVE_COMPLEX_H
  compspec->data[0] = REAL(s->specdata[0]);
  for (i = 1; i < s->fft_size -1 ; i++) {
    compspec->data[i] = REAL(s->specdata[i]);
    compspec->data[compspec->length - i] = IMAG(s->specdata[i]);
  }
  compspec->data[s->fft_size-1] = REAL(s->specdata[s->fft_size-1]);
#else /* HAVE_COMPLEX_H  */
  for (i = 0; i < s->fft_size; i++) {
    compspec->data[i] = s->specdata[i];
  }
#endif /* HAVE_COMPLEX_H */

#elif defined HAVE_ACCELERATE // using ACCELERATE
  // convert real data to even/odd format used in vDSP
  aubio_vDSP_ctoz((aubio_DSPComplex*)s->in, 2, &s->spec, 1, s->fft_size/2);
  // compute the FFT
  aubio_vDSP_fft_zrip(s->fftSetup, &s->spec, 1, s->log2fftsize, FFT_FORWARD);
  // convert from vDSP complex split to [ r0, r1, ..., rN, iN-1, .., i2, i1]
  compspec->data[0] = s->spec.realp[0];
  compspec->data[s->fft_size / 2] = s->spec.imagp[0];
  for (i = 1; i < s->fft_size / 2; i++) {
    compspec->data[i] = s->spec.realp[i];
    compspec->data[s->fft_size - i] = s->spec.imagp[i];
  }
  // apply scaling
  smpl_t scale = 1./2.;
  aubio_vDSP_vsmul(compspec->data, 1, &scale, compspec->data, 1, s->fft_size);

#elif defined HAVE_INTEL_IPP  // using Intel IPP

  // apply fft
#if HAVE_AUBIO_DOUBLE
  ippsFFTFwd_RToCCS_64f(s->in, (Ipp64f*)s->complexOut, s->fftSpec, s->memBuffer);
#else
  ippsFFTFwd_RToCCS_32f(s->in, (Ipp32f*)s->complexOut, s->fftSpec, s->memBuffer);
#endif
  // convert complex buffer to [ r0, r1, ..., rN, iN-1, .., i2, i1]
  compspec->data[0] = s->complexOut[0].re;
  compspec->data[s->fft_size / 2] = s->complexOut[s->fft_size / 2].re;
  for (i = 1; i < s->fft_size / 2; i++) {
    compspec->data[i] = s->complexOut[i].re;
    compspec->data[s->fft_size - i] = s->complexOut[i].im;
  }
  // apply scaling
#if HAVE_AUBIO_DOUBLE
  ippsMulC_64f(compspec->data, 1.0 / 2.0, compspec->data, s->fft_size);
#else
  ippsMulC_32f(compspec->data, 1.0 / 2.0, compspec->data, s->fft_size);
#endif

#else                         // using OOURA
  aubio_ooura_rdft(s->winsize, 1, s->in, s->ip, s->w);
  compspec->data[0] = s->in[0];
  compspec->data[s->winsize / 2] = s->in[1];
  for (i = 1; i < s->fft_size - 1; i++) {
    compspec->data[i] = s->in[2 * i];
    compspec->data[s->winsize - i] = - s->in[2 * i + 1];
  }
#endif /* using OOURA */
}

void aubio_fft_rdo_complex(aubio_fft_t * s, const fvec_t * compspec, fvec_t * output) {
  uint_t i;
#ifdef HAVE_FFTW3
  const smpl_t renorm = 1./(smpl_t)s->winsize;
#ifdef HAVE_COMPLEX_H
  s->specdata[0] = compspec->data[0];
  for (i=1; i < s->fft_size - 1; i++) {
    s->specdata[i] = compspec->data[i] +
      I * compspec->data[compspec->length - i];
  }
  s->specdata[s->fft_size - 1] = compspec->data[s->fft_size - 1];
#else
  for (i=0; i < s->fft_size; i++) {
    s->specdata[i] = compspec->data[i];
  }
#endif
  fftw_execute(s->pbw);
  for (i = 0; i < output->length; i++) {
    output->data[i] = s->out[i]*renorm;
  }

#elif defined HAVE_ACCELERATE // using ACCELERATE
  // convert from real imag  [ r0, r1, ..., rN, iN-1, .., i2, i1]
  // to vDSP packed format   [ r0, rN, r1, i1, ..., rN-1, iN-1 ]
  s->out[0] = compspec->data[0];
  s->out[1] = compspec->data[s->winsize / 2];
  for (i = 1; i < s->fft_size / 2; i++) {
    s->out[2 * i] = compspec->data[i];
    s->out[2 * i + 1] = compspec->data[s->winsize - i];
  }
  // convert to split complex format used in vDSP
  aubio_vDSP_ctoz((aubio_DSPComplex*)s->out, 2, &s->spec, 1, s->fft_size/2);
  // compute the FFT
  aubio_vDSP_fft_zrip(s->fftSetup, &s->spec, 1, s->log2fftsize, FFT_INVERSE);
  // convert result to real output
  aubio_vDSP_ztoc(&s->spec, 1, (aubio_DSPComplex*)output->data, 2, s->fft_size/2);
  // apply scaling
  smpl_t scale = 1.0 / s->winsize;
  aubio_vDSP_vsmul(output->data, 1, &scale, output->data, 1, s->fft_size);

#elif defined HAVE_INTEL_IPP  // using Intel IPP

  // convert from real imag  [ r0, 0, ..., rN, iN-1, .., i2, i1, iN-1] to complex format
  s->complexOut[0].re = compspec->data[0];
  s->complexOut[0].im = 0;
  s->complexOut[s->fft_size / 2].re = compspec->data[s->fft_size / 2];
  s->complexOut[s->fft_size / 2].im = 0.0;
  for (i = 1; i < s->fft_size / 2; i++) {
    s->complexOut[i].re = compspec->data[i];
    s->complexOut[i].im = compspec->data[s->fft_size - i];
  }
#if HAVE_AUBIO_DOUBLE
  // apply fft
  ippsFFTInv_CCSToR_64f((const Ipp64f *)s->complexOut, output->data, s->fftSpec, s->memBuffer);
  // apply scaling
  ippsMulC_64f(output->data, 1.0 / s->winsize, output->data, s->fft_size);
#else
  // apply fft
  ippsFFTInv_CCSToR_32f((const Ipp32f *)s->complexOut, output->data, s->fftSpec, s->memBuffer);
  // apply scaling
  ippsMulC_32f(output->data, 1.0f / s->winsize, output->data, s->fft_size);
#endif /* HAVE_AUBIO_DOUBLE */

#else                         // using OOURA
  smpl_t scale = 1.0 / s->winsize;
  s->out[0] = compspec->data[0];
  s->out[1] = compspec->data[s->winsize / 2];
  for (i = 1; i < s->fft_size - 1; i++) {
    s->out[2 * i] = compspec->data[i];
    s->out[2 * i + 1] = - compspec->data[s->winsize - i];
  }
  aubio_ooura_rdft(s->winsize, -1, s->out, s->ip, s->w);
  for (i=0; i < s->winsize; i++) {
    output->data[i] = s->out[i] * scale;
  }
#endif
}

void aubio_fft_get_spectrum(aubio_fft_t *s, const fvec_t * compspec, cvec_t * spectrum) {
  aubio_fft_get_phas(s, compspec, spectrum);
  aubio_fft_get_norm(s, compspec, spectrum);
}

void aubio_fft_get_realimag(aubio_fft_t *s, const cvec_t * spectrum, fvec_t * compspec) {
  aubio_fft_get_imag(s, spectrum, compspec);
  aubio_fft_get_real(s, spectrum, compspec);
}

void aubio_fft_get_phas(aubio_fft_t *s, const fvec_t * compspec, cvec_t * spectrum) {

#ifdef INTEL_IPP_FFT // using Intel IPP FFT
  uint_t i;
  
  // convert from real imag  [ r0, 0, ..., rN, iN-1, .., i2, i1, iN-1] to complex format
  s->complexOut[0].re = compspec->data[0];
  s->complexOut[0].im = 0;
  s->complexOut[s->fft_size / 2].re = compspec->data[s->fft_size / 2];
  s->complexOut[s->fft_size / 2].im = 0.0;
  for (i = 1; i < spectrum->length - 1; i++) {
    s->complexOut[i].re = compspec->data[i];
    s->complexOut[i].im = compspec->data[compspec->length - i];
  }
  
#if HAVE_AUBIO_DOUBLE
  IppStatus status = ippsPhase_64fc(s->complexOut, spectrum->phas, spectrum->length);
#else
  IppStatus status = ippsPhase_32fc(s->complexOut, spectrum->phas, spectrum->length);
#endif
  if (status != ippStsNoErr) {
    AUBIO_ERR("fft: failed to extract phase from fft. IPP error: %d\n", status);
  }

#else                 // NOT using Intel IPP
  uint_t i;
  if (compspec->data[0] < 0) {
    spectrum->phas[0] = PI;
  } else {
    spectrum->phas[0] = 0.;
  }
  for (i=1; i < spectrum->length - 1; i++) {
    spectrum->phas[i] = ATAN2(compspec->data[compspec->length-i],
        compspec->data[i]);
  }
  if (compspec->data[compspec->length/2] < 0) {
    spectrum->phas[spectrum->length - 1] = PI;
  } else {
    spectrum->phas[spectrum->length - 1] = 0.;
  }
#endif
}

void aubio_fft_get_norm(aubio_fft_t *s, const fvec_t * compspec, cvec_t * spectrum) {
  uint_t i = 0;
  spectrum->norm[0] = ABS(compspec->data[0]);
  for (i=1; i < spectrum->length - 1; i++) {
    spectrum->norm[i] = SQRT(SQR(compspec->data[i])
        + SQR(compspec->data[compspec->length - i]) );
  }
  spectrum->norm[spectrum->length-1] =
    ABS(compspec->data[compspec->length/2]);
}

void aubio_fft_get_imag(aubio_fft_t *s, const cvec_t * spectrum, fvec_t * compspec) {
  uint_t i;
  for (i = 1; i < ( compspec->length + 1 ) / 2 /*- 1 + 1*/; i++) {
    compspec->data[compspec->length - i] =
      spectrum->norm[i]*SIN(spectrum->phas[i]);
  }
}

void aubio_fft_get_real(aubio_fft_t *s, const cvec_t * spectrum, fvec_t * compspec) {
  uint_t i;
  for (i = 0; i < compspec->length / 2 + 1; i++) {
    compspec->data[i] =
      spectrum->norm[i]*COS(spectrum->phas[i]);
  }
}

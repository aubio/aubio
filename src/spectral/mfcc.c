/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>
                      and Amaury Hazan <ahazan@iua.upf.edu>

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
#include "fmat.h"
#include "cvec.h"
#include "mathutils.h"
#include "vecutils.h"
#include "spectral/fft.h"
#include "spectral/filterbank.h"
#include "spectral/filterbank_mel.h"
#include "spectral/dct.h"
#include "spectral/mfcc.h"

#ifdef HAVE_NOOPT
#define HAVE_SLOW_DCT 1
#endif

/** Internal structure for mfcc object */

struct _aubio_mfcc_t
{
  uint_t win_s;             /** grain length */
  uint_t samplerate;        /** sample rate (needed?) */
  uint_t n_filters;         /** number of filters */
  uint_t n_coefs;           /** number of coefficients (<= n_filters/2 +1) */
  aubio_filterbank_t *fb;   /** filter bank */
  fvec_t *in_dct;           /** input buffer for dct * [fb->n_filters] */
#if defined(HAVE_SLOW_DCT)
  fmat_t *dct_coeffs;       /** DCT transform n_filters * n_coeffs */
#else
  aubio_dct_t *dct;
  fvec_t *output;
#endif
  smpl_t scale;
};


aubio_mfcc_t *
new_aubio_mfcc (uint_t win_s, uint_t n_filters, uint_t n_coefs,
    uint_t samplerate)
{

  /* allocate space for mfcc object */
  aubio_mfcc_t *mfcc = AUBIO_NEW (aubio_mfcc_t);
#if defined(HAVE_SLOW_DCT)
  smpl_t scaling;

  uint_t i, j;
#endif

  mfcc->win_s = win_s;
  mfcc->samplerate = samplerate;
  mfcc->n_filters = n_filters;
  mfcc->n_coefs = n_coefs;

  /* filterbank allocation */
  mfcc->fb = new_aubio_filterbank (n_filters, mfcc->win_s);
  aubio_filterbank_set_mel_coeffs_slaney (mfcc->fb, samplerate);

  /* allocating buffers */
  mfcc->in_dct = new_fvec (n_filters);

#if defined(HAVE_SLOW_DCT)
  mfcc->dct_coeffs = new_fmat (n_coefs, n_filters);

  /* compute DCT transform dct_coeffs[j][i] as
     cos ( j * (i+.5) * PI / n_filters ) */
  scaling = 1. / SQRT (n_filters / 2.);
  for (i = 0; i < n_filters; i++) {
    for (j = 0; j < n_coefs; j++) {
      mfcc->dct_coeffs->data[j][i] =
          scaling * COS (j * (i + 0.5) * PI / n_filters);
    }
    mfcc->dct_coeffs->data[0][i] *= SQRT (2.) / 2.;
  }
#else
  mfcc->dct = new_aubio_dct (n_filters);
  mfcc->output = new_fvec (n_filters);
#endif

  mfcc->scale = 1.;

  return mfcc;
}

void
del_aubio_mfcc (aubio_mfcc_t * mf)
{

  /* delete filterbank */
  del_aubio_filterbank (mf->fb);

  /* delete buffers */
  del_fvec (mf->in_dct);
#if defined(HAVE_SLOW_DCT)
  del_fmat (mf->dct_coeffs);
#else
  del_aubio_dct (mf->dct);
  del_fvec (mf->output);
#endif

  /* delete mfcc object */
  AUBIO_FREE (mf);
}


void
aubio_mfcc_do (aubio_mfcc_t * mf, const cvec_t * in, fvec_t * out)
{
#ifndef HAVE_SLOW_DCT
  fvec_t tmp;
#endif

  /* compute filterbank */
  aubio_filterbank_do (mf->fb, in, mf->in_dct);

  /* compute log10 */
  fvec_log10 (mf->in_dct);

  if (mf->scale != 1) fvec_mul (mf->in_dct, mf->scale);

  /* compute mfccs */
#if defined(HAVE_SLOW_DCT)
  fmat_vecmul(mf->dct_coeffs, mf->in_dct, out);
#else
  aubio_dct_do(mf->dct, mf->in_dct, mf->output);
  // copy only first n_coeffs elements
  // TODO assert mf->output->length == n_coeffs
  tmp.data = mf->output->data;
  tmp.length = out->length;
  fvec_copy(&tmp, out);
#endif

  return;
}

uint_t aubio_mfcc_set_power (aubio_mfcc_t *mf, smpl_t power)
{
  return aubio_filterbank_set_power(mf->fb, power);
}

uint_t aubio_mfcc_get_power (aubio_mfcc_t *mf)
{
  return aubio_filterbank_get_power(mf->fb);
}

uint_t aubio_mfcc_set_scale (aubio_mfcc_t *mf, smpl_t scale)
{
  mf->scale = scale;
  return AUBIO_OK;
}

uint_t aubio_mfcc_get_scale (aubio_mfcc_t *mf)
{
  return mf->scale;
}

uint_t aubio_mfcc_set_mel_coeffs (aubio_mfcc_t *mf, smpl_t freq_min,
    smpl_t freq_max)
{
  return aubio_filterbank_set_mel_coeffs(mf->fb, mf->samplerate,
      freq_min, freq_max);
}

uint_t aubio_mfcc_set_mel_coeffs_htk (aubio_mfcc_t *mf, smpl_t freq_min,
    smpl_t freq_max)
{
  return aubio_filterbank_set_mel_coeffs_htk(mf->fb, mf->samplerate,
      freq_min, freq_max);
}

uint_t aubio_mfcc_set_mel_coeffs_slaney (aubio_mfcc_t *mf, smpl_t freq_min,
    smpl_t freq_max)
{
  return aubio_filterbank_set_mel_coeffs_slaney (mf->fb, mf->samplerate);
}

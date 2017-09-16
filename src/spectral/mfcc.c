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
#include "spectral/mfcc.h"

/** Internal structure for mfcc object */

struct _aubio_mfcc_t
{
  uint_t win_s;             /** grain length */
  uint_t samplerate;        /** sample rate (needed?) */
  uint_t n_filters;         /** number of filters */
  uint_t n_coefs;           /** number of coefficients (<= n_filters/2 +1) */
  aubio_filterbank_t *fb;   /** filter bank */
  fvec_t *in_dct;           /** input buffer for dct * [fb->n_filters] */
  fmat_t *dct_coeffs;       /** DCT transform n_filters * n_coeffs */
};


aubio_mfcc_t *
new_aubio_mfcc (uint_t win_s, uint_t n_filters, uint_t n_coefs,
    uint_t samplerate)
{

  /* allocate space for mfcc object */
  aubio_mfcc_t *mfcc = AUBIO_NEW (aubio_mfcc_t);
  smpl_t scaling;

  uint_t i, j;

  mfcc->win_s = win_s;
  mfcc->samplerate = samplerate;
  mfcc->n_filters = n_filters;
  mfcc->n_coefs = n_coefs;

  /* filterbank allocation */
  mfcc->fb = new_aubio_filterbank (n_filters, mfcc->win_s);
  aubio_filterbank_set_mel_coeffs_slaney (mfcc->fb, samplerate);

  /* allocating buffers */
  mfcc->in_dct = new_fvec (n_filters);

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

  return mfcc;
}

void
del_aubio_mfcc (aubio_mfcc_t * mf)
{

  /* delete filterbank */
  del_aubio_filterbank (mf->fb);

  /* delete buffers */
  del_fvec (mf->in_dct);
  del_fmat (mf->dct_coeffs);

  /* delete mfcc object */
  AUBIO_FREE (mf);
}


void
aubio_mfcc_do (aubio_mfcc_t * mf, const cvec_t * in, fvec_t * out)
{
  /* compute filterbank */
  aubio_filterbank_do (mf->fb, in, mf->in_dct);

  /* compute log10 */
  fvec_log10 (mf->in_dct);

  /* raise power */
  //fvec_pow (mf->in_dct, 3.);

  /* compute mfccs */
  fmat_vecmul(mf->dct_coeffs, mf->in_dct, out);

  return;
}

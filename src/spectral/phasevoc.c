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
#include "spectral/phasevoc.h"

/** phasevocoder internal object */
struct _aubio_pvoc_t {
  uint_t win_s;       /** grain length */
  uint_t hop_s;       /** overlap step */
  aubio_fft_t * fft;  /** fft object */
  fvec_t * synth;     /** cur output grain [win_s] */
  fvec_t * synthold;  /** last input frame [win_s-hop_s] */
  fvec_t * data;      /** current input grain [win_s] */
  fvec_t * dataold;   /** last input frame [win_s-hop_s] */
  fvec_t * w;         /** grain window [win_s] */
};


/** returns data and dataold slided by hop_s */
static void aubio_pvoc_swapbuffers(smpl_t * data, smpl_t * dataold, const
    smpl_t * datanew, uint_t win_s, uint_t hop_s);

/** do additive synthesis from 'old' and 'cur' */
static void aubio_pvoc_addsynth(const smpl_t * synth, smpl_t * synthold,
    smpl_t * synthnew, uint_t win_s, uint_t hop_s);

void aubio_pvoc_do(aubio_pvoc_t *pv, fvec_t * datanew, cvec_t *fftgrain) {
  /* slide  */
  aubio_pvoc_swapbuffers(pv->data->data,pv->dataold->data,
      datanew->data,pv->win_s,pv->hop_s);
  /* windowing */
  fvec_weight(pv->data, pv->w);
  /* shift */
  fvec_shift(pv->data);
  /* calculate fft */
  aubio_fft_do (pv->fft,pv->data,fftgrain);
}

void aubio_pvoc_rdo(aubio_pvoc_t *pv,cvec_t * fftgrain, fvec_t * synthnew) {
  /* calculate rfft */
  aubio_fft_rdo(pv->fft,fftgrain,pv->synth);
  /* unshift */
  fvec_shift(pv->synth);
  aubio_pvoc_addsynth(pv->synth->data,pv->synthold->data,
      synthnew->data,pv->win_s,pv->hop_s);
}

aubio_pvoc_t * new_aubio_pvoc (uint_t win_s, uint_t hop_s) {
  aubio_pvoc_t * pv = AUBIO_NEW(aubio_pvoc_t);

  /* if (win_s < 2*hop_s) {
    AUBIO_WRN("Hop size bigger than half the window size!\n");
  } */

  if (hop_s < 1) {
    AUBIO_ERR("got hop_size %d, but can not be < 1\n", hop_s);
    goto beach;
  } else if (win_s < 1) {
    AUBIO_ERR("got buffer_size %d, but can not be < 2\n", win_s);
    goto beach;
  } else if (win_s < hop_s + 1) {
    AUBIO_ERR("hop size (%d) is larger than or equal to win size (%d)\n", win_s, hop_s);
    goto beach;
  }

  pv->fft      = new_aubio_fft (win_s);

  /* remember old */
  pv->data     = new_fvec (win_s);
  pv->synth    = new_fvec (win_s);

  /* new input output */
  pv->dataold  = new_fvec  (win_s-hop_s);
  pv->synthold = new_fvec (win_s-hop_s);
  pv->w        = new_aubio_window ("hanningz", win_s);

  pv->hop_s    = hop_s;
  pv->win_s    = win_s;

  return pv;

beach:
  AUBIO_FREE (pv);
  return NULL;
}

void del_aubio_pvoc(aubio_pvoc_t *pv) {
  del_fvec(pv->data);
  del_fvec(pv->synth);
  del_fvec(pv->dataold);
  del_fvec(pv->synthold);
  del_fvec(pv->w);
  del_aubio_fft(pv->fft);
  AUBIO_FREE(pv);
}

static void aubio_pvoc_swapbuffers(smpl_t * data, smpl_t * dataold, 
    const smpl_t * datanew, uint_t win_s, uint_t hop_s)
{
  uint_t i;
  for (i = 0; i < win_s - hop_s; i++)
    data[i] = dataold[i];
  for (i = 0; i < hop_s; i++)
    data[win_s - hop_s + i] = datanew[i];
  for (i = 0; i < win_s - hop_s; i++)
    dataold[i] = data[i + hop_s];
}

static void aubio_pvoc_addsynth(const smpl_t * synth, smpl_t * synthold, 
                smpl_t * synthnew, uint_t win_s, uint_t hop_s)
{
  uint_t i;
  smpl_t scale = 2 * hop_s / (win_s + .0);
  /* add new synth to old one and put result in synthnew */
  for (i = 0; i < hop_s; i++)
    synthnew[i] = synthold[i] + synth[i] * scale;
  /* shift synthold */
  for (i = 0; i < win_s - 2 * hop_s; i++)
    synthold[i] = synthold[i + hop_s];
  /* erase last frame in synthold */
  for (i = win_s - hop_s; i < win_s; i++)
    synthold[i - hop_s] = 0.;
  /* additive synth */
  for (i = 0; i < win_s - hop_s; i++)
    synthold[i] += synth[i + hop_s] * scale;
}


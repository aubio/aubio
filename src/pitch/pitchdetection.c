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

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "spectral/phasevoc.h"
#include "mathutils.h"
#include "temporal/filter.h"
#include "pitch/pitchmcomb.h"
#include "pitch/pitchyin.h"
#include "pitch/pitchfcomb.h"
#include "pitch/pitchschmitt.h"
#include "pitch/pitchyinfft.h"
#include "pitch/pitchdetection.h"

typedef smpl_t (*aubio_pitchdetection_func_t)
  (aubio_pitchdetection_t *p, fvec_t * ibuf);
typedef smpl_t (*aubio_pitchdetection_conv_t)
  (smpl_t value, uint_t srate, uint_t bufsize);

void aubio_pitchdetection_slideblock(aubio_pitchdetection_t *p, fvec_t *ibuf);

smpl_t aubio_pitchdetection_mcomb   (aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_yin     (aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_schmitt (aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_fcomb   (aubio_pitchdetection_t *p, fvec_t *ibuf);
smpl_t aubio_pitchdetection_yinfft  (aubio_pitchdetection_t *p, fvec_t *ibuf);

/** generic pitch detection structure */
struct _aubio_pitchdetection_t {
  aubio_pitchdetection_type type; /**< pitch detection mode */
  aubio_pitchdetection_mode mode; /**< pitch detection output mode */
  uint_t srate;                   /**< samplerate */
  uint_t bufsize;                 /**< buffer size */
  aubio_pitchmcomb_t * mcomb;     /**< mcomb object */
  aubio_pitchfcomb_t * fcomb;     /**< fcomb object */
  aubio_pitchschmitt_t * schmitt; /**< schmitt object */
  aubio_pitchyinfft_t * yinfft;   /**< yinfft object */
  aubio_filter_t * filter;        /**< filter */
  aubio_pvoc_t * pv;              /**< phase vocoder for mcomb */ 
  cvec_t * fftgrain;              /**< spectral frame for mcomb */
  fvec_t * buf;                   /**< temporary buffer for yin */
  fvec_t * yin;                   /**< yin function */
  smpl_t yinthres;                /**< yin peak picking threshold parameter */
  aubio_pitchdetection_func_t callback; /**< pointer to current pitch detection method */
  aubio_pitchdetection_conv_t freqconv; /**< pointer to current pitch conversion method */ 
};

/* convenience wrapper function for frequency unit conversions 
 * should probably be rewritten with #defines */
smpl_t freqconvbin(smpl_t f,uint_t srate,uint_t bufsize);
smpl_t freqconvbin(smpl_t f,uint_t srate,uint_t bufsize){
  return aubio_freqtobin(f,srate,bufsize);
}

smpl_t freqconvmidi(smpl_t f,uint_t srate,uint_t bufsize);
smpl_t freqconvmidi(smpl_t f,uint_t srate UNUSED,uint_t bufsize UNUSED){
  return aubio_freqtomidi(f);
}

smpl_t freqconvpass(smpl_t f,uint_t srate,uint_t bufsize);
smpl_t freqconvpass(smpl_t f,uint_t srate UNUSED,uint_t bufsize UNUSED){
  return f;
}

aubio_pitchdetection_t * new_aubio_pitchdetection(uint_t bufsize, 
    uint_t hopsize, 
    uint_t channels,
    uint_t samplerate,
    aubio_pitchdetection_type type,
    aubio_pitchdetection_mode mode)
{
  aubio_pitchdetection_t *p = AUBIO_NEW(aubio_pitchdetection_t);
  p->srate = samplerate;
  p->type = type;
  p->mode = mode;
  p->bufsize = bufsize;
  switch(p->type) {
    case aubio_pitch_yin:
      p->buf      = new_fvec(bufsize,channels);
      p->yin      = new_fvec(bufsize/2,channels);
      p->callback = aubio_pitchdetection_yin;
      p->yinthres = 0.15;
      break;
    case aubio_pitch_mcomb:
      p->pv       = new_aubio_pvoc(bufsize, hopsize, channels);
      p->fftgrain = new_cvec(bufsize, channels);
      p->mcomb    = new_aubio_pitchmcomb(bufsize,hopsize,channels,samplerate);
      p->filter   = new_aubio_cdsgn_filter(samplerate);
      p->callback = aubio_pitchdetection_mcomb;
      break;
    case aubio_pitch_fcomb:
      p->buf      = new_fvec(bufsize,channels);
      p->fcomb    = new_aubio_pitchfcomb(bufsize,hopsize,samplerate);
      p->callback = aubio_pitchdetection_fcomb;
      break;
    case aubio_pitch_schmitt:
      p->buf      = new_fvec(bufsize,channels);
      p->schmitt  = new_aubio_pitchschmitt(bufsize,samplerate);
      p->callback = aubio_pitchdetection_schmitt;
      break;
    case aubio_pitch_yinfft:
      p->buf      = new_fvec(bufsize,channels);
      p->yinfft   = new_aubio_pitchyinfft(bufsize);
      p->callback = aubio_pitchdetection_yinfft;
      p->yinthres = 0.85;
      break;
    default:
      break;
  }
  switch(p->mode) {
    case aubio_pitchm_freq:
      p->freqconv = freqconvpass;
      break;
    case aubio_pitchm_midi:
      p->freqconv = freqconvmidi;
      break;
    case aubio_pitchm_cent:
      /* bug: not implemented */
      p->freqconv = freqconvmidi;
      break;
    case aubio_pitchm_bin:
      p->freqconv = freqconvbin;
      break;
    default:
      break;
  }
  return p;
}

void del_aubio_pitchdetection(aubio_pitchdetection_t * p) {
  switch(p->type) {
    case aubio_pitch_yin:
      del_fvec(p->yin);
      del_fvec(p->buf);
      break;
    case aubio_pitch_mcomb:
      del_aubio_pvoc(p->pv);
      del_cvec(p->fftgrain);
      del_aubio_filter(p->filter);
      del_aubio_pitchmcomb(p->mcomb);
      break;
    case aubio_pitch_schmitt:
      del_fvec(p->buf);
      del_aubio_pitchschmitt(p->schmitt);
      break;
    case aubio_pitch_fcomb:
      del_fvec(p->buf);
      del_aubio_pitchfcomb(p->fcomb);
      break;
    case aubio_pitch_yinfft:
      del_fvec(p->buf);
      del_aubio_pitchyinfft(p->yinfft);
      break;
    default:
      break;
  }
  AUBIO_FREE(p);
}

void aubio_pitchdetection_slideblock(aubio_pitchdetection_t *p, fvec_t *ibuf){
  uint_t i,j = 0, overlap_size = 0;
  overlap_size = p->buf->length-ibuf->length;
  for (i=0;i<p->buf->channels;i++){
    for (j=0;j<overlap_size;j++){
      p->buf->data[i][j] = p->buf->data[i][j+ibuf->length];
    }
  }
  for (i=0;i<ibuf->channels;i++){
    for (j=0;j<ibuf->length;j++){
      p->buf->data[i][j+overlap_size] = ibuf->data[i][j];
    }
  }
}

void aubio_pitchdetection_set_yinthresh(aubio_pitchdetection_t *p, smpl_t thres) {
  p->yinthres = thres;
}

smpl_t aubio_pitchdetection(aubio_pitchdetection_t *p, fvec_t * ibuf) {
  return p->freqconv(p->callback(p,ibuf),p->srate,p->bufsize);
}

smpl_t aubio_pitchdetection_mcomb(aubio_pitchdetection_t *p, fvec_t *ibuf) {
  smpl_t pitch = 0.;
  aubio_filter_do(p->filter,ibuf);
  aubio_pvoc_do(p->pv,ibuf,p->fftgrain);
  pitch = aubio_pitchmcomb_detect(p->mcomb,p->fftgrain);
  /** \bug should move the >0 check within aubio_bintofreq */
  if (pitch>0.) {
    pitch = aubio_bintofreq(pitch,p->srate,p->bufsize);
  } else {
    pitch = 0.;
  }
  return pitch;
}

smpl_t aubio_pitchdetection_yin(aubio_pitchdetection_t *p, fvec_t *ibuf) {
  smpl_t pitch = 0.;
  aubio_pitchdetection_slideblock(p,ibuf);
  pitch = aubio_pitchyin_getpitchfast(p->buf,p->yin, p->yinthres);
  if (pitch>0) {
    pitch = p->srate/(pitch+0.);
  } else {
    pitch = 0.;
  }
  return pitch;
}


smpl_t aubio_pitchdetection_yinfft(aubio_pitchdetection_t *p, fvec_t *ibuf){
  smpl_t pitch = 0.;
  aubio_pitchdetection_slideblock(p,ibuf);
  pitch = aubio_pitchyinfft_detect(p->yinfft,p->buf,p->yinthres);
  if (pitch>0) {
    pitch = p->srate/(pitch+0.);
  } else {
    pitch = 0.;
  }
  return pitch; 
}

smpl_t aubio_pitchdetection_fcomb(aubio_pitchdetection_t *p, fvec_t *ibuf){
  aubio_pitchdetection_slideblock(p,ibuf);
  return aubio_pitchfcomb_detect(p->fcomb,p->buf);
}

smpl_t aubio_pitchdetection_schmitt(aubio_pitchdetection_t *p, fvec_t *ibuf){
  aubio_pitchdetection_slideblock(p,ibuf);
  return aubio_pitchschmitt_detect(p->schmitt,p->buf);
}

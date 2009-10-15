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
#include "lvec.h"
#include "spectral/phasevoc.h"
#include "mathutils.h"
#include "temporal/filter.h"
#include "temporal/c_weighting.h"
#include "pitch/pitchmcomb.h"
#include "pitch/pitchyin.h"
#include "pitch/pitchfcomb.h"
#include "pitch/pitchschmitt.h"
#include "pitch/pitchyinfft.h"
#include "pitch/pitchdetection.h"

/** pitch detection algorithm */
typedef enum {
  aubio_pitch_yin,     /**< YIN algorithm */
  aubio_pitch_mcomb,   /**< Multi-comb filter */
  aubio_pitch_schmitt, /**< Schmitt trigger */
  aubio_pitch_fcomb,   /**< Fast comb filter */
  aubio_pitch_yinfft,   /**< Spectral YIN */
  aubio_pitch_default = aubio_pitch_yinfft, /**< the one used when "default" is asked */
} aubio_pitchdetection_type;

/** pitch detection output mode */
typedef enum {
  aubio_pitchm_freq,   /**< Frequency (Hz) */
  aubio_pitchm_midi,   /**< MIDI note (0.,127) */
  aubio_pitchm_cent,   /**< Cent */
  aubio_pitchm_bin,    /**< Frequency bin (0,bufsize) */
  aubio_pitchm_default = aubio_pitchm_freq, /**< the one used when "default" is asked */
} aubio_pitchdetection_mode;

typedef void (*aubio_pitchdetection_func_t)
  (aubio_pitchdetection_t *p, fvec_t * ibuf, fvec_t *obuf);
typedef smpl_t (*aubio_pitchdetection_conv_t)
  (smpl_t value, uint_t srate, uint_t bufsize);

void aubio_pitchdetection_slideblock(aubio_pitchdetection_t *p, fvec_t *ibuf);

void aubio_pitchdetection_mcomb   (aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *obuf);
void aubio_pitchdetection_yin     (aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *obuf);
void aubio_pitchdetection_schmitt (aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *obuf);
void aubio_pitchdetection_fcomb   (aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *obuf);
void aubio_pitchdetection_yinfft  (aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *obuf);

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
  aubio_pitchyin_t * yin;   /**< yinfft object */
  aubio_filter_t * filter;        /**< filter */
  aubio_pvoc_t * pv;              /**< phase vocoder for mcomb */ 
  cvec_t * fftgrain;              /**< spectral frame for mcomb */
  fvec_t * buf;                   /**< temporary buffer for yin */
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

aubio_pitchdetection_t *
new_aubio_pitchdetection (char_t * pitch_mode,
    uint_t bufsize, uint_t hopsize, uint_t channels, uint_t samplerate)
{
  aubio_pitchdetection_t *p = AUBIO_NEW(aubio_pitchdetection_t);
  aubio_pitchdetection_type pitch_type;
  if (strcmp (pitch_mode, "mcomb") == 0)
      pitch_type = aubio_pitch_mcomb;
  else if (strcmp (pitch_mode, "yinfft") == 0)
      pitch_type = aubio_pitch_yin;
  else if (strcmp (pitch_mode, "yin") == 0)
      pitch_type = aubio_pitch_yin;
  else if (strcmp (pitch_mode, "schmitt") == 0)
      pitch_type = aubio_pitch_schmitt;
  else if (strcmp (pitch_mode, "fcomb") == 0)
      pitch_type = aubio_pitch_fcomb;
  else if (strcmp (pitch_mode, "default") == 0)
      pitch_type = aubio_pitch_default;
  else {
      AUBIO_ERR ("unknown pitch detection method %s, using default.\n", pitch_mode);
      pitch_type = aubio_pitch_default;
      return NULL;
  }
  p->srate = samplerate;
  p->type = pitch_type;
  aubio_pitchdetection_set_unit (p, "default");
  p->bufsize = bufsize;
  switch(p->type) {
    case aubio_pitch_yin:
      p->buf      = new_fvec(bufsize,channels);
      p->yin      = new_aubio_pitchyin(bufsize);
      p->callback = aubio_pitchdetection_yin;
      aubio_pitchyin_set_tolerance (p->yin, 0.15);
      break;
    case aubio_pitch_mcomb:
      p->pv       = new_aubio_pvoc(bufsize, hopsize, channels);
      p->fftgrain = new_cvec(bufsize, channels);
      p->mcomb    = new_aubio_pitchmcomb(bufsize,hopsize,channels);
      p->filter   = new_aubio_filter_c_weighting (samplerate, channels);
      p->callback = aubio_pitchdetection_mcomb;
      break;
    case aubio_pitch_fcomb:
      p->buf      = new_fvec(bufsize,channels);
      p->fcomb    = new_aubio_pitchfcomb(bufsize,hopsize,channels);
      p->callback = aubio_pitchdetection_fcomb;
      break;
    case aubio_pitch_schmitt:
      p->buf      = new_fvec(bufsize,channels);
      p->schmitt  = new_aubio_pitchschmitt(bufsize);
      p->callback = aubio_pitchdetection_schmitt;
      break;
    case aubio_pitch_yinfft:
      p->buf      = new_fvec(bufsize,channels);
      p->yinfft   = new_aubio_pitchyinfft(bufsize);
      p->callback = aubio_pitchdetection_yinfft;
      aubio_pitchyinfft_set_tolerance (p->yinfft, 0.85);
      break;
    default:
      break;
  }
  return p;
}

void del_aubio_pitchdetection(aubio_pitchdetection_t * p) {
  switch(p->type) {
    case aubio_pitch_yin:
      del_fvec(p->buf);
      del_aubio_pitchyin(p->yin);
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

uint_t aubio_pitchdetection_set_unit (aubio_pitchdetection_t *p, char_t * pitch_unit) {
  aubio_pitchdetection_mode pitch_mode;
  if (strcmp (pitch_unit, "freq") == 0)
      pitch_mode = aubio_pitchm_freq;
  else if (strcmp (pitch_unit, "midi") == 0)
      pitch_mode = aubio_pitchm_midi;
  else if (strcmp (pitch_unit, "cent") == 0)
      pitch_mode = aubio_pitchm_cent;
  else if (strcmp (pitch_unit, "bin") == 0)
      pitch_mode = aubio_pitchm_bin;
  else if (strcmp (pitch_unit, "default") == 0)
      pitch_mode = aubio_pitchm_default;
  else {
      AUBIO_ERR ("unknown pitch detection unit %s, using default\n", pitch_unit);
      pitch_mode = aubio_pitchm_default;
  }
  p->mode = pitch_mode;
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
  return AUBIO_OK;
}

uint_t aubio_pitchdetection_set_tolerance(aubio_pitchdetection_t *p, smpl_t tol) {
  switch(p->type) {
    case aubio_pitch_yin:
      aubio_pitchyin_set_tolerance (p->yin, tol);
      break;
    case aubio_pitch_yinfft:
      aubio_pitchyinfft_set_tolerance (p->yinfft, tol);
      break;
    default:
      break;
  }
  return AUBIO_OK;
}

void aubio_pitchdetection_do (aubio_pitchdetection_t *p, fvec_t * ibuf, fvec_t *obuf) {
  uint_t i;
  p->callback(p, ibuf, obuf);
  for (i = 0; i < obuf->channels; i++) {
    p->freqconv(obuf->data[i][0],p->srate,p->bufsize);
  }
}

void aubio_pitchdetection_mcomb(aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t * obuf) {
  uint_t i;
  aubio_filter_do(p->filter,ibuf);
  aubio_pvoc_do(p->pv,ibuf,p->fftgrain);
  aubio_pitchmcomb_do(p->mcomb,p->fftgrain, obuf);
  for (i = 0; i < obuf->channels; i++) {
    obuf->data[i][0] = aubio_bintofreq (obuf->data[i][0], p->srate, p->bufsize);
  }
}

void aubio_pitchdetection_yin(aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t * obuf) {
  smpl_t pitch = 0.;
  uint_t i;
  aubio_pitchdetection_slideblock(p,ibuf);
  aubio_pitchyin_do(p->yin,p->buf, obuf);
  for (i = 0; i < obuf->channels; i++) {
    pitch = obuf->data[i][0];
    if (pitch>0) {
      pitch = p->srate/(pitch+0.);
    } else {
      pitch = 0.;
    }
    obuf->data[i][0] = pitch;
  }
}


void aubio_pitchdetection_yinfft(aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t * obuf){
  smpl_t pitch = 0.;
  uint_t i;
  aubio_pitchdetection_slideblock(p,ibuf);
  aubio_pitchyinfft_do(p->yinfft,p->buf,obuf);
  for (i = 0; i < obuf->channels; i++) {
    pitch = obuf->data[i][0];
    if (pitch>0) {
      pitch = p->srate/(pitch+0.);
    } else {
      pitch = 0.;
    }
    obuf->data[i][0] = pitch;
  }
}

void aubio_pitchdetection_fcomb(aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t * out){
  uint_t i;
  aubio_pitchdetection_slideblock(p,ibuf);
  aubio_pitchfcomb_do(p->fcomb,p->buf, out);
  for (i = 0; i < out->channels; i++) {
    out->data[i][0] = aubio_bintofreq (out->data[i][0], p->srate, p->bufsize);
  }
}

void aubio_pitchdetection_schmitt(aubio_pitchdetection_t *p, fvec_t *ibuf, fvec_t *out){
  smpl_t period, pitch = 0.;
  uint_t i;
  aubio_pitchdetection_slideblock(p,ibuf);
  aubio_pitchschmitt_do(p->schmitt,p->buf, out);
  for (i = 0; i < out->channels; i++) {
    period = out->data[i][0];
    if (period>0) {
      pitch = p->srate/period;
    } else {
      pitch = 0.;
    }
    out->data[i][0] = pitch;
  }
}

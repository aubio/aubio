/*
   Copyright (C) 2006 Paul Brossier

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
#include "onset/onsetdetection.h"
#include "spectral/phasevoc.h"
#include "onset/peakpick.h"
#include "mathutils.h"
#include "onset/onset.h"

/** structure to store object state */
struct _aubio_onset_t {
  aubio_pvoc_t * pv;            /**< phase vocoder */
  aubio_onsetdetection_t * od;  /**< onset detection */ 
  aubio_peakpicker_t * pp;      /**< peak picker */
  cvec_t * fftgrain;            /**< phase vocoder output */
  fvec_t * of;                  /**< onset detection function */
  smpl_t threshold;             /**< onset peak picking threshold */
  smpl_t silence;               /**< silence threhsold */
  uint_t minioi;                /**< minimum inter onset interval */
  uint_t wasonset;              /**< number of frames since last onset */
  uint_t samplerate;            /**< sampling rate of the input signal */
};

/* execute onset detection function on iput buffer */
void aubio_onset_do (aubio_onset_t *o, fvec_t * input, fvec_t * onset)
{
  uint_t isonset = 0;
  uint_t wasonset = o->wasonset;
  aubio_pvoc_do (o->pv,input, o->fftgrain);
  aubio_onsetdetection_do (o->od,o->fftgrain, o->of);
  /*if (usedoubled) {
    aubio_onsetdetection_do (o2,fftgrain, onset2);
    onset->data[0][0] *= onset2->data[0][0];
  }*/
  isonset = aubio_peakpicker_do(o->pp, o->of);
  if (isonset > 0.) {
    if (aubio_silence_detection(input, o->silence)==1) {
      isonset  = 0;
      wasonset++;
    } else {
      if (wasonset > o->minioi) {
        wasonset = 0;
      } else {
        isonset  = 0;
        wasonset++;
      }
    }
  } else {
    wasonset++;
  }
  o->wasonset = wasonset;
  onset->data[0][0] = isonset;
  return;
}

uint_t aubio_onset_set_silence(aubio_onset_t * o, smpl_t silence) {
  o->silence = silence;
  return AUBIO_OK;
}

uint_t aubio_onset_set_threshold(aubio_onset_t * o, smpl_t threshold) {
  o->threshold = threshold;
  aubio_peakpicker_set_threshold(o->pp, o->threshold);
  return AUBIO_OK;
}

uint_t aubio_onset_set_minioi(aubio_onset_t * o, uint_t minioi) {
  o->minioi = minioi;
  return AUBIO_OK;
}

/* Allocate memory for an onset detection */
aubio_onset_t * new_aubio_onset (char_t * onset_mode, 
    uint_t buf_size, uint_t hop_size, uint_t channels, uint_t samplerate)
{
  aubio_onset_t * o = AUBIO_NEW(aubio_onset_t);
  /** set some default parameter */
  o->threshold = 0.3;
  o->minioi    = 4;
  o->silence   = -70;
  o->wasonset  = 0;
  o->samplerate = samplerate;
  o->pv = new_aubio_pvoc(buf_size, hop_size, channels);
  o->pp = new_aubio_peakpicker(o->threshold);
  o->od = new_aubio_onsetdetection(onset_mode,buf_size,channels);
  o->fftgrain = new_cvec(buf_size,channels);
  o->of = new_fvec(1, channels);
  /*if (usedoubled)    {
    o2 = new_aubio_onsetdetection(onset_type2,buffer_size,channels);
    onset2 = new_fvec(1 , channels);
  }*/
  return o;
}

void del_aubio_onset (aubio_onset_t *o)
{
  del_aubio_onsetdetection(o->od);
  del_aubio_peakpicker(o->pp);
  del_aubio_pvoc(o->pv);
  del_fvec(o->of);
  del_cvec(o->fftgrain);
  AUBIO_FREE(o);
}

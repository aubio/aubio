/*
  Copyright (C) 2006-2013 Paul Brossier <piem@aubio.org>

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
#include "spectral/specdesc.h"
#include "spectral/phasevoc.h"
#include "onset/peakpicker.h"
#include "mathutils.h"
#include "onset/onset.h"

/** structure to store object state */
struct _aubio_onset_t {
  aubio_pvoc_t * pv;            /**< phase vocoder */
  aubio_specdesc_t * od;        /**< spectral descriptor */
  aubio_peakpicker_t * pp;      /**< peak picker */
  cvec_t * fftgrain;            /**< phase vocoder output */
  fvec_t * desc;                /**< spectral description */
  smpl_t silence;               /**< silence threhsold */
  uint_t minioi;                /**< minimum inter onset interval */
  uint_t delay;                 /**< constant delay, in samples, removed from detected onset times */
  uint_t samplerate;            /**< sampling rate of the input signal */
  uint_t hop_size;              /**< number of samples between two runs */

  uint_t total_frames;          /**< total number of frames processed since the beginning */
  uint_t last_onset;            /**< last detected onset location, in frames */
};

/* execute onset detection function on iput buffer */
void aubio_onset_do (aubio_onset_t *o, fvec_t * input, fvec_t * onset)
{
  smpl_t isonset = 0;
  aubio_pvoc_do (o->pv,input, o->fftgrain);
  aubio_specdesc_do (o->od, o->fftgrain, o->desc);
  aubio_peakpicker_do(o->pp, o->desc, onset);
  isonset = onset->data[0];
  if (isonset > 0.) {
    if (aubio_silence_detection(input, o->silence)==1) {
      //AUBIO_DBG ("silent onset, not marking as onset\n");
      isonset  = 0;
    } else {
      uint_t new_onset = o->total_frames + (uint_t)ROUND(isonset * o->hop_size);
      if (o->last_onset + o->minioi < new_onset) {
        //AUBIO_DBG ("accepted detection, marking as onset\n");
        o->last_onset = new_onset;
      } else {
        //AUBIO_DBG ("doubled onset, not marking as onset\n");
        isonset  = 0;
      }
    }
  } else {
    // we are at the beginning of the file, and we don't find silence
    if (o->total_frames <= o->delay && o->last_onset < o ->minioi && aubio_silence_detection(input, o->silence) == 0) {
      //AUBIO_DBG ("beginning of file is not silent, marking as onset\n");
      isonset = o->delay / o->hop_size;
      o->last_onset = o->delay;
    }
  }
  onset->data[0] = isonset;
  o->total_frames += o->hop_size;
  return;
}

uint_t aubio_onset_get_last (aubio_onset_t *o)
{
  return o->last_onset - o->delay;
}

smpl_t aubio_onset_get_last_s (aubio_onset_t *o)
{
  return aubio_onset_get_last (o) / (smpl_t) (o->samplerate);
}

smpl_t aubio_onset_get_last_ms (aubio_onset_t *o)
{
  return aubio_onset_get_last_s (o) * 1000.;
}

uint_t aubio_onset_set_silence(aubio_onset_t * o, smpl_t silence) {
  o->silence = silence;
  return AUBIO_OK;
}

smpl_t aubio_onset_get_silence(aubio_onset_t * o) {
  return o->silence;
}

uint_t aubio_onset_set_threshold(aubio_onset_t * o, smpl_t threshold) {
  aubio_peakpicker_set_threshold(o->pp, threshold);
  return AUBIO_OK;
}

smpl_t aubio_onset_get_threshold(aubio_onset_t * o) {
  return aubio_peakpicker_get_threshold(o->pp);
}

uint_t aubio_onset_set_minioi(aubio_onset_t * o, uint_t minioi) {
  o->minioi = minioi;
  return AUBIO_OK;
}

uint_t aubio_onset_get_minioi(aubio_onset_t * o) {
  return o->minioi;
}

uint_t aubio_onset_set_minioi_s(aubio_onset_t * o, smpl_t minioi) {
  return aubio_onset_set_minioi (o, minioi * o->samplerate);
}

smpl_t aubio_onset_get_minioi_s(aubio_onset_t * o) {
  return aubio_onset_get_minioi (o) / (smpl_t) o->samplerate;
}

uint_t aubio_onset_set_minioi_ms(aubio_onset_t * o, smpl_t minioi) {
  return aubio_onset_set_minioi_s (o, minioi / 1000.);
}

smpl_t aubio_onset_get_minioi_ms(aubio_onset_t * o) {
  return aubio_onset_get_minioi_s (o) * 1000.;
}

uint_t aubio_onset_set_delay(aubio_onset_t * o, uint_t delay) {
  o->delay = delay;
  return AUBIO_OK;
}

uint_t aubio_onset_get_delay(aubio_onset_t * o) {
  return o->delay;
}

uint_t aubio_onset_set_delay_s(aubio_onset_t * o, smpl_t delay) {
  return aubio_onset_set_delay (o, delay * o->samplerate);
}

smpl_t aubio_onset_get_delay_s(aubio_onset_t * o) {
  return aubio_onset_get_delay (o) / (smpl_t) o->samplerate;
}

uint_t aubio_onset_set_delay_ms(aubio_onset_t * o, smpl_t delay) {
  return aubio_onset_set_delay_s (o, delay / 1000.);
}

smpl_t aubio_onset_get_delay_ms(aubio_onset_t * o) {
  return aubio_onset_get_delay_s (o) * 1000.;
}

smpl_t aubio_onset_get_descriptor(aubio_onset_t * o) {
  return o->desc->data[0];
}

smpl_t aubio_onset_get_thresholded_descriptor(aubio_onset_t * o) {
  fvec_t * thresholded = aubio_peakpicker_get_thresholded_input(o->pp);
  return thresholded->data[0];
}

/* Allocate memory for an onset detection */
aubio_onset_t * new_aubio_onset (char_t * onset_mode, 
    uint_t buf_size, uint_t hop_size, uint_t samplerate)
{
  aubio_onset_t * o = AUBIO_NEW(aubio_onset_t);

  /* check parameters are valid */
  if ((sint_t)hop_size < 1) {
    AUBIO_ERR("onset: got hop_size %d, but can not be < 1\n", hop_size);
    goto beach;
  } else if ((sint_t)buf_size < 1) {
    AUBIO_ERR("onset: got buffer_size %d, but can not be < 1\n", buf_size);
    goto beach;
  } else if (buf_size < hop_size) {
    AUBIO_ERR("onset: hop size (%d) is larger than win size (%d)\n", buf_size, hop_size);
    goto beach;
  } else if ((sint_t)samplerate < 1) {
    AUBIO_ERR("onset: samplerate (%d) can not be < 1\n", samplerate);
    goto beach;
  }

  /* store creation parameters */
  o->samplerate = samplerate;
  o->hop_size = hop_size;

  /* allocate memory */
  o->pv = new_aubio_pvoc(buf_size, o->hop_size);
  o->pp = new_aubio_peakpicker();
  o->od = new_aubio_specdesc(onset_mode,buf_size);
  o->fftgrain = new_cvec(buf_size);
  o->desc = new_fvec(1);

  /* set some default parameter */
  aubio_onset_set_threshold (o, 0.3);
  aubio_onset_set_delay(o, 4.3 * hop_size);
  aubio_onset_set_minioi_ms(o, 20.);
  aubio_onset_set_silence(o, -70.);

  /* initialize internal variables */
  o->last_onset = 0;
  o->total_frames = 0;
  return o;

beach:
  AUBIO_FREE(o);
  return NULL;
}

void del_aubio_onset (aubio_onset_t *o)
{
  del_aubio_specdesc(o->od);
  del_aubio_peakpicker(o->pp);
  del_aubio_pvoc(o->pv);
  del_fvec(o->desc);
  del_cvec(o->fftgrain);
  AUBIO_FREE(o);
}

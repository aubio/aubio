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
#include "sample.h"
#include "onsetdetection.h"
#include "beattracking.h"
#include "phasevoc.h"
#include "peakpick.h"
#include "mathutils.h"
#include "tempo.h"

/* structure to store object state */
struct _aubio_tempo_t {
  aubio_onsetdetection_t * od;   /** onset detection */
  aubio_pvoc_t * pv;             /** phase vocoder */
  aubio_pickpeak_t * pp;         /** peak picker */
  aubio_beattracking_t * bt;     /** beat tracking */
  cvec_t * fftgrain;             /** spectral frame */
  fvec_t * of;                   /** onset detection function value */
  fvec_t * dfframe;              /** peak picked detection function buffer */
  fvec_t * out;                  /** beat tactus candidates */
  smpl_t silence;                /** silence parameter */
  smpl_t threshold;              /** peak picking threshold */
  sint_t blockpos;               /** current position in dfframe */
  uint_t winlen;                 /** dfframe bufsize */
  uint_t step;                   /** dfframe hopsize */ 
};

/* execute tempo detection function on iput buffer */
void aubio_tempo(aubio_tempo_t *o, fvec_t * input, fvec_t * tempo)
{
  uint_t i;
  uint_t winlen = o->winlen;
  uint_t step   = o->step;
  aubio_pvoc_do (o->pv, input, o->fftgrain);
  aubio_onsetdetection(o->od, o->fftgrain, o->of);
  /*if (usedoubled) {
    aubio_onsetdetection(o2,fftgrain, onset2);
    onset->data[0][0] *= onset2->data[0][0];
  }*/
  /* execute every overlap_size*step */
  if (o->blockpos == (signed)step -1 ) {
    /* check dfframe */
    aubio_beattracking_do(o->bt,o->dfframe,o->out);
    /* rotate dfframe */
    for (i = 0 ; i < winlen - step; i++ ) 
      o->dfframe->data[0][i] = o->dfframe->data[0][i+step];
    for (i = winlen - step ; i < winlen; i++ ) 
      o->dfframe->data[0][i] = 0.;
    o->blockpos = -1;
  }
  o->blockpos++;
  tempo->data[0][1] = aubio_peakpick_pimrt_wt(o->of,o->pp,
    &(o->dfframe->data[0][winlen - step + o->blockpos]));
  /* end of second level loop */
  tempo->data[0][0] = 0; /* reset tactus */
  i=0;
  for (i = 1; i < o->out->data[0][0]; i++ ) {
    /* if current frame is a predicted tactus */
    if (o->blockpos == o->out->data[0][i]) {
      /* test for silence */
      if (aubio_silence_detection(input, o->silence)==1) {
        tempo->data[0][1] = 0; /* unset onset */
        tempo->data[0][0] = 0; /* unset tactus */
      } else {
        tempo->data[0][0] = 1; /* set tactus */
      }
    }
  }
}

void aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence) {
  o->silence = silence;
  return;
}

void aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold) {
  o->threshold = threshold;
  aubio_peakpicker_set_threshold(o->pp, o->threshold);
  return;
}

/* Allocate memory for an tempo detection */
aubio_tempo_t * new_aubio_tempo (aubio_onsetdetection_type type_onset, 
    uint_t buf_size, uint_t hop_size, uint_t channels)
{
  aubio_tempo_t * o = AUBIO_NEW(aubio_tempo_t);
  o->winlen = SQR(512)/hop_size;
  o->step = o->winlen/4;
  o->blockpos = 0;
  o->threshold = 0.3;
  o->silence = -90;
  o->blockpos = 0;
  o->dfframe  = new_fvec(o->winlen,channels);
  o->fftgrain = new_cvec(buf_size, channels);
  o->out      = new_fvec(o->step,channels);
  o->pv       = new_aubio_pvoc(buf_size, hop_size, channels);
  o->pp       = new_aubio_peakpicker(o->threshold);
  o->od       = new_aubio_onsetdetection(type_onset,buf_size,channels);
  o->of       = new_fvec(1, channels);
  o->bt       = new_aubio_beattracking(o->winlen,channels);
  /*if (usedoubled)    {
    o2 = new_aubio_onsetdetection(type_onset2,buffer_size,channels);
    onset2 = new_fvec(1 , channels);
  }*/
  return o;
}

void del_aubio_tempo (aubio_tempo_t *o)
{
  del_aubio_onsetdetection(o->od);
  del_aubio_beattracking(o->bt);
  del_aubio_peakpicker(o->pp);
  del_aubio_pvoc(o->pv);
  del_fvec(o->out);
  del_fvec(o->of);
  del_cvec(o->fftgrain);
  del_fvec(o->dfframe);
  AUBIO_FREE(o);
  return;
}

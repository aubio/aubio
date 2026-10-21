/*
  Copyright (C) 2006-2009 Paul Brossier <piem@aubio.org>

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
#include "tempo/beattracking.h"
#include "spectral/phasevoc.h"
#include "onset/peakpicker.h"
#include "mathutils.h"
#include "tempo/tempo.h"

/* structure to store object state */
struct _aubio_tempo_t {
  aubio_specdesc_t * od;   /** onset detection */
  aubio_pvoc_t * pv;             /** phase vocoder */
  aubio_peakpicker_t * pp;       /** peak picker */
  aubio_beattracking_t * bt;     /** beat tracking */
  cvec_t * fftgrain;             /** spectral frame */
  fvec_t * of;                   /** onset detection function value */
  fvec_t * dfframe;              /** peak picked detection function buffer */
  fvec_t * out;                  /** beat tactus candidates */
  fvec_t * onset;                /** onset results */
  smpl_t silence;                /** silence parameter */
  smpl_t threshold;              /** peak picking threshold */
  sint_t blockpos;               /** current position in dfframe */
  uint_t winlen;                 /** dfframe bufsize */
  uint_t step;                   /** dfframe hopsize */
  uint_t samplerate;             /** sampling rate of the signal */
  uint_t hop_size;               /** get hop_size */
  uint_t total_frames;           /** total frames since beginning */
  uint_t last_beat;              /** time of latest detected beat, in samples */
  sint_t delay;                  /** delay to remove to last beat, in samples */
  smpl_t latency;                /** analysis latency taken back, in frames */
  uint_t last_tatum;             /** time of latest detected tatum, in samples */
  uint_t tatum_signature;        /** number of tatum between each beats */
};

/* Latency of the detection function, in samples, as
 *   window * min(buf_size / 4, 6 * hop_size) + hop * hop_size
 * A spectral frame scores an attack highest about a quarter of the window
 * after it, but the peak picker smooths over a fixed number of frames, so
 * with a window of more than 24 hops the peak comes about 6 hops in. How
 * sharply each function rises sets the two weights. Fitted, per function,
 * on drums at 100 to 160 bpm, 22050 to 48000Hz, hops of 64 to 1024 and
 * windows of 2 to 64 hops; checked at 110 to 170 bpm. Up to windows of
 * 250 ms and hops of 30 ms, beats are within 2 ms on average, 6 ms for 95%
 * of the settings. Longer windows hold several hits, and the latency
 * depends on the music more than on the analysis. */
typedef struct {
  const char_t *method;
  double window;
  double hop;
} aubio_tempo_latency_t;

static const aubio_tempo_latency_t aubio_tempo_latencies[] = {
  { "specflux", 1.01, 0.69 },   /* also "default", and any other */
  { "energy",   2.02, 0.60 },
  { "hfc",      1.37, 0.94 },
  { "complex",  0.68, 1.55 },
  { "phase",    0.29, 1.37 },
  { "wphase",   0.42, 1.45 },
  { "specdiff", 1.14, 1.32 },
  { "kl",       1.32, 0.61 },
  { "mkl",      0.67, 0.84 },
};

static smpl_t aubio_tempo_latency (const char_t *method, uint_t buf_size,
    uint_t hop_size)
{
  const aubio_tempo_latency_t *l = &aubio_tempo_latencies[0];
  uint_t i;
  for (i = 0; i < sizeof (aubio_tempo_latencies)
      / sizeof (aubio_tempo_latencies[0]); i++) {
    if (strcmp (method, aubio_tempo_latencies[i].method) == 0) {
      l = &aubio_tempo_latencies[i];
    }
  }
  return (smpl_t)(l->window * MIN (buf_size / 4., 6. * hop_size)
    + l->hop * hop_size);
}

/* The beats predicted for a step are where the detection function will
 * show them, which is later than they are heard, by the latency above. Each
 * beat is fired that much earlier, so the beats predicted up to that much
 * past the end of the step are added here: the next step's own prediction
 * only starts at its beginning. */
static void aubio_tempo_extend_beats (aubio_tempo_t *o)
{
  uint_t n = (uint_t)o->out->data[0];
  smpl_t bp = aubio_beattracking_get_period (o->bt) / (smpl_t)o->hop_size;
  smpl_t beat;
  if (n < 2 || bp <= 0) return;
  beat = o->out->data[n - 1];
  while (beat + bp < (smpl_t)o->step + o->latency && n < o->out->length) {
    beat += bp;
    o->out->data[n++] = beat;
  }
  o->out->data[0] = (smpl_t)n;
}

/* execute tempo detection function on iput buffer */
void aubio_tempo_do(aubio_tempo_t *o, const fvec_t * input, fvec_t * tempo)
{
  uint_t i;
  uint_t winlen = o->winlen;
  uint_t step   = o->step;
  fvec_t * thresholded;
  aubio_pvoc_do (o->pv, input, o->fftgrain);
  aubio_specdesc_do (o->od, o->fftgrain, o->of);
  /*if (usedoubled) {
    aubio_specdesc_do(o2,fftgrain, onset2);
    onset->data[0] *= onset2->data[0];
  }*/
  /* execute every overlap_size*step */
  if (o->blockpos == (signed)step -1 ) {
    /* check dfframe */
    aubio_beattracking_do(o->bt,o->dfframe,o->out);
    aubio_tempo_extend_beats(o);
    /* rotate dfframe */
    for (i = 0 ; i < winlen - step; i++ )
      o->dfframe->data[i] = o->dfframe->data[i+step];
    for (i = winlen - step ; i < winlen; i++ )
      o->dfframe->data[i] = 0.;
    o->blockpos = -1;
  }
  o->blockpos++;
  aubio_peakpicker_do (o->pp, o->of, o->onset);
  // store onset detection function in second sample of vector
  //tempo->data[1] = o->onset->data[0];
  thresholded = aubio_peakpicker_get_thresholded_input(o->pp);
  o->dfframe->data[winlen - step + o->blockpos] = thresholded->data[0];
  /* end of second level loop */
  tempo->data[0] = 0; /* reset tactus */
  //i=0;
  for (i = 1; (smpl_t)i < o->out->data[0]; i++ ) {
    /* the beat happened latency frames before it was predicted */
    smpl_t beat = o->out->data[i] - o->latency;
    /* if current frame is a predicted tactus */
    if (beat >= 0 && (smpl_t)o->blockpos == FLOOR(beat)) {
      tempo->data[0] = beat - FLOOR(beat); /* set tactus */
      /* test for silence */
      if (aubio_silence_detection(input, o->silence)==1) {
        tempo->data[0] = 0; // unset beat if silent
      }
      o->last_beat = o->total_frames + (uint_t)ROUND(tempo->data[0] * (smpl_t)o->hop_size);
      o->last_tatum = o->last_beat;
    }
  }
  o->total_frames += o->hop_size;
  return;
}

uint_t aubio_tempo_get_last (aubio_tempo_t *o)
{
  return o->last_beat + o->delay;
}

smpl_t aubio_tempo_get_last_s (aubio_tempo_t *o)
{
  return (smpl_t)aubio_tempo_get_last (o) / (smpl_t)o->samplerate;
}

smpl_t aubio_tempo_get_last_ms (aubio_tempo_t *o)
{
  return aubio_tempo_get_last_s (o) * 1000;
}

uint_t aubio_tempo_set_delay(aubio_tempo_t * o, sint_t delay) {
  o->delay = delay;
  return AUBIO_OK;
}

uint_t aubio_tempo_set_delay_s(aubio_tempo_t * o, smpl_t delay) {
  o->delay = (sint_t)ROUND(delay * (smpl_t)o->samplerate);
  return AUBIO_OK;
}

uint_t aubio_tempo_set_delay_ms(aubio_tempo_t * o, smpl_t delay) {
  return aubio_tempo_set_delay_s(o, delay / 1000);
}

sint_t aubio_tempo_get_delay(aubio_tempo_t * o) {
  return o->delay;
}

smpl_t aubio_tempo_get_delay_s(aubio_tempo_t * o) {
  return (smpl_t)o->delay / (smpl_t)o->samplerate;
}

smpl_t aubio_tempo_get_delay_ms(aubio_tempo_t * o) {
  return aubio_tempo_get_delay_s(o) * 1000;
}

uint_t aubio_tempo_set_silence(aubio_tempo_t * o, smpl_t silence) {
  o->silence = silence;
  return AUBIO_OK;
}

smpl_t aubio_tempo_get_silence(aubio_tempo_t * o) {
  return o->silence;
}

uint_t aubio_tempo_set_threshold(aubio_tempo_t * o, smpl_t threshold) {
  o->threshold = threshold;
  aubio_peakpicker_set_threshold(o->pp, o->threshold);
  return AUBIO_OK;
}

smpl_t aubio_tempo_get_threshold(aubio_tempo_t * o) {
  return o->threshold;
}

/* Allocate memory for an tempo detection */
aubio_tempo_t * new_aubio_tempo (const char_t * tempo_mode,
    uint_t buf_size, uint_t hop_size, uint_t samplerate)
{
  aubio_tempo_t * o = AUBIO_NEW(aubio_tempo_t);
  char_t specdesc_func[PATH_MAX];

  if (!o) {
    return NULL;
  }

  o->samplerate = samplerate;
  // check parameters are valid
  if ((sint_t)hop_size < 1) {
    AUBIO_ERR("tempo: got hop size %d, but can not be < 1\n", hop_size);
    goto beach;
  } else if ((sint_t)buf_size < 2) {
    AUBIO_ERR("tempo: got window size %d, but can not be < 2\n", buf_size);
    goto beach;
  } else if (buf_size < hop_size) {
    AUBIO_ERR("tempo: hop size (%d) is larger than window size (%d)\n", buf_size, hop_size);
    goto beach;
  } else if ((sint_t)samplerate < 1) {
    AUBIO_ERR("tempo: samplerate (%d) can not be < 1\n", samplerate);
    goto beach;
  }

  /* length of observations, worth about 6 seconds: 512 frames at 44100Hz
   * with a hop of 512, the setting the beat tracker was tuned at, and the
   * same time at any other. It was the next power of two of 5.8 s, which
   * is anything from 5.8 to 11.6 s: 8.5 s at 30000/500, and as the beat
   * tracker runs every quarter of it, a tempo change was followed every
   * 2.1 s instead of 1.5 s. A multiple of 4, for step and laglen. */
  o->winlen = 4 * (uint_t)floor(512. / 4. * 512. / 44100.
      * samplerate / hop_size + .5);
  if (o->winlen < 4) o->winlen = 4;
  o->step = o->winlen/4;
  o->blockpos = 0;
  o->threshold = (smpl_t)0.3;
  o->silence = -90.;
  o->total_frames = 0;
  o->last_beat = 0;
  o->delay = 0;
  o->hop_size = hop_size;
  o->dfframe  = new_fvec(o->winlen);
  o->fftgrain = new_cvec(buf_size);
  o->out      = new_fvec(o->step);
  o->pv       = new_aubio_pvoc(buf_size, hop_size);
  o->pp       = new_aubio_peakpicker();
  aubio_peakpicker_set_threshold (o->pp, o->threshold);
  if ( strcmp(tempo_mode, "default") == 0 ) {
    strncpy(specdesc_func, "specflux", PATH_MAX - 1);
    specdesc_func[PATH_MAX - 1] = '\0';
  } else {
    strncpy(specdesc_func, tempo_mode, PATH_MAX - 1);
    specdesc_func[PATH_MAX - 1] = '\0';
  }
  o->od       = new_aubio_specdesc(specdesc_func,buf_size);
  o->latency  = aubio_tempo_latency(specdesc_func, buf_size, hop_size)
    / (smpl_t)hop_size;
  o->of       = new_fvec(1);
  o->bt       = new_aubio_beattracking(o->winlen, o->hop_size, o->samplerate);
  o->onset    = new_fvec(1);
  /*if (usedoubled)    {
    o2 = new_aubio_specdesc(type_onset2,buffer_size);
    onset2 = new_fvec(1);
  }*/
  if (!o->dfframe || !o->fftgrain || !o->out || !o->pv ||
      !o->pp || !o->od || !o->of || !o->bt || !o->onset) {
    AUBIO_ERR("tempo: failed creating tempo object\n");
    goto beach;
  }
  o->last_tatum = 0;
  o->tatum_signature = 4;
  return o;

beach:
  del_aubio_tempo(o);
  return NULL;
}

smpl_t aubio_tempo_get_bpm(aubio_tempo_t *o) {
  return aubio_beattracking_get_bpm(o->bt);
}

smpl_t aubio_tempo_get_period (aubio_tempo_t *o)
{
  return aubio_beattracking_get_period (o->bt);
}

smpl_t aubio_tempo_get_period_s (aubio_tempo_t *o)
{
  return aubio_beattracking_get_period_s (o->bt);
}

smpl_t aubio_tempo_get_confidence(aubio_tempo_t *o) {
  return aubio_beattracking_get_confidence(o->bt);
}

uint_t aubio_tempo_was_tatum (aubio_tempo_t *o)
{
  uint_t last_tatum_distance = o->total_frames - o->last_tatum;
  smpl_t beat_period = aubio_tempo_get_period(o);
  smpl_t tatum_period = beat_period / (smpl_t)o->tatum_signature;
  if (last_tatum_distance < o->hop_size) {
    o->last_tatum = o->last_beat;
    return 2;
  }
  else if ((smpl_t)last_tatum_distance > tatum_period) {
    if ( (smpl_t)(last_tatum_distance + o->hop_size) > beat_period ) {
      // next beat is too close, pass
      return 0;
    }
    o->last_tatum = o->total_frames;
    return 1;
  }
  return 0;
}

smpl_t aubio_tempo_get_last_tatum (aubio_tempo_t *o) {
  return (smpl_t)o->last_tatum - (smpl_t)o->delay;
}

uint_t aubio_tempo_set_tatum_signature (aubio_tempo_t *o, uint_t signature) {
  if (signature < 1 || signature > 64) {
    return AUBIO_FAIL;
  } else {
    o->tatum_signature = signature;
    return AUBIO_OK;
  }
}

void del_aubio_tempo (aubio_tempo_t *o)
{
  if (o->od)
    del_aubio_specdesc(o->od);
  if (o->bt)
    del_aubio_beattracking(o->bt);
  if (o->pp)
    del_aubio_peakpicker(o->pp);
  if (o->pv)
    del_aubio_pvoc(o->pv);
  if (o->out)
    del_fvec(o->out);
  if (o->of)
    del_fvec(o->of);
  if (o->fftgrain)
    del_cvec(o->fftgrain);
  if (o->dfframe)
    del_fvec(o->dfframe);
  if (o->onset)
    del_fvec(o->onset);
  AUBIO_FREE(o);
}

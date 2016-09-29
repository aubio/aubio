/*
  Copyright (C) 2016 Paul Brossier <piem@aubio.org>

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

#include "config.h"

#ifdef HAVE_RUBBERBAND

#include "aubio_priv.h"
#include "fvec.h"
#include "fmat.h"
#include "io/source.h"
#include "effects/timestretch.h"

#include "rubberband/rubberband-c.h"

#define MIN_STRETCH_RATIO 0.025
#define MAX_STRETCH_RATIO 40.

/** generic time stretching structure */
struct _aubio_timestretch_t
{
  uint_t samplerate;              /**< samplerate */
  uint_t hopsize;                 /**< hop size */
  smpl_t stretchratio;            /**< time ratio */
  smpl_t pitchscale;              /**< pitch scale */

  aubio_source_t *source;
  fvec_t *in;
  fvec_t *zeros;
  uint_t eof;

  RubberBandState rb;
  RubberBandOptions rboptions;
};

extern RubberBandOptions aubio_get_rubberband_opts(const char_t *mode);

aubio_timestretch_t *
new_aubio_timestretch (const char_t * uri, const char_t * mode,
    smpl_t stretchratio, uint_t hopsize, uint_t samplerate)
{
  aubio_timestretch_t *p = AUBIO_NEW (aubio_timestretch_t);
  p->samplerate = samplerate;
  p->hopsize = hopsize;
  p->pitchscale = 1.;
  p->eof = 0;

  p->source = new_aubio_source(uri, samplerate, hopsize);
  if (!p->source) goto beach;
  if (samplerate == 0 ) p->samplerate = aubio_source_get_samplerate(p->source);

  p->in = new_fvec(hopsize);
  p->zeros = new_fvec(hopsize);

  if (stretchratio <= MAX_STRETCH_RATIO && stretchratio >= MIN_STRETCH_RATIO) {
    p->stretchratio = stretchratio;
  } else {
    AUBIO_ERR("timestretch: stretchratio should be in the range [%.3f, %.3f], got %f\n",
        MIN_STRETCH_RATIO, MAX_STRETCH_RATIO, stretchratio);
    goto beach;
  }

  p->rboptions = aubio_get_rubberband_opts(mode);
  if (p->rboptions < 0) {
    AUBIO_ERR("timestretch: unknown time stretching method %s\n", mode);
    goto beach;
  }

  p->rb = rubberband_new(p->samplerate, 1, p->rboptions, p->stretchratio, p->pitchscale);
  rubberband_set_max_process_size(p->rb, p->hopsize);
  //rubberband_set_debug_level(p->rb, 10);

#if 1
  // warm up rubber band
  uint_t source_read = 0;
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
  int available = rubberband_available(p->rb);
  while (available <= (int)latency) {
    aubio_source_do(p->source, p->in, &source_read);
    // for very short samples
    if (source_read < p->hopsize) p->eof = 1;
    rubberband_process(p->rb, (const float* const*)&(p->in->data), p->hopsize, p->eof);
    available = rubberband_available(p->rb);
  }
#endif

  return p;

beach:
  del_aubio_timestretch(p);
  return NULL;
}

void
del_aubio_timestretch (aubio_timestretch_t * p)
{
  if (p->source) del_aubio_source(p->source);
  if (p->rb) {
    rubberband_delete(p->rb);
  }
  AUBIO_FREE (p);
}

uint_t
aubio_timestretch_get_samplerate (aubio_timestretch_t * p)
{
  return p->samplerate;
}

uint_t aubio_timestretch_get_latency (aubio_timestretch_t * p) {
  return rubberband_get_latency(p->rb);
}

uint_t
aubio_timestretch_set_stretch (aubio_timestretch_t * p, smpl_t stretch)
{
  if (stretch >= MIN_STRETCH_RATIO && stretch <= MAX_STRETCH_RATIO) {
    p->stretchratio = stretch;
    rubberband_set_time_ratio(p->rb, 1./p->stretchratio);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("timestretch: could not set stretch ratio to %.2f\n", stretch);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_timestretch_get_stretch (aubio_timestretch_t * p)
{
  return p->stretchratio;
}

uint_t
aubio_timestretch_set_pitchscale (aubio_timestretch_t * p, smpl_t pitchscale)
{
  if (pitchscale >= 0.0625  && pitchscale <= 4.) {
    p->pitchscale = pitchscale;
    rubberband_set_pitch_scale(p->rb, p->pitchscale);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("timestretch: could not set pitchscale to %.2f\n", pitchscale);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_timestretch_get_pitchscale (aubio_timestretch_t * p)
{
  return p->pitchscale;
}

uint_t
aubio_timestretch_set_transpose(aubio_timestretch_t * p, smpl_t transpose)
{
  if (transpose >= -24. && transpose <= 24.) {
    smpl_t pitchscale = POW(2., transpose / 12.);
    return aubio_timestretch_set_pitchscale(p, pitchscale);
  } else {
    AUBIO_ERR("timestretch: could not set transpose to %.2f\n", transpose);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_timestretch_get_transpose(aubio_timestretch_t * p)
{
  return 12. * LOG(p->pitchscale) / LOG(2.0);
}

void
aubio_timestretch_do (aubio_timestretch_t * p, fvec_t * out, uint_t * read)
{
  uint_t source_read = p->hopsize;
  // read more samples from source until we have enough available or eof is reached
  int available = rubberband_available(p->rb);
  while ((available < (int)p->hopsize) && (p->eof == 0)) {
    aubio_source_do(p->source, p->in, &source_read);
    if (source_read < p->hopsize) {
      p->eof = 1;
    }
    rubberband_process(p->rb, (const float* const*)&(p->in->data), source_read, p->eof);
    available = rubberband_available(p->rb);
  }
  // now retrieve the samples and write them into out->data
  if (available >= (int)p->hopsize) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), p->hopsize);
    *read = p->hopsize;
  } else {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), available);
    *read = available;
  }
}

uint_t
aubio_timestretch_seek (aubio_timestretch_t *p, uint_t pos)
{
  p->eof = 0;
  rubberband_reset(p->rb);
  return aubio_source_seek(p->source, pos);
}

#endif

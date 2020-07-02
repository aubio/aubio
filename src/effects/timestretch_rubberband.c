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

#include "aubio_priv.h"

#ifdef HAVE_RUBBERBAND

#include "fvec.h"
#include "fmat.h"
#include "io/source.h"
#include "effects/timestretch.h"

#include <rubberband/rubberband-c.h>

#define MIN_STRETCH_RATIO 0.025
#define MAX_STRETCH_RATIO 40.

#define HAVE_THREADS 1
#if 0
#undef HAVE_THREADS
#endif

#ifdef HAVE_THREADS
#include <pthread.h>
#endif

/** generic time stretching structure */
struct _aubio_timestretch_t
{
  uint_t samplerate;              /**< samplerate */
  uint_t hopsize;                 /**< hop size */
  smpl_t stretchratio;            /**< time ratio */
  smpl_t pitchscale;              /**< pitch scale */

  RubberBandState rb;
  RubberBandOptions rboptions;
};

extern RubberBandOptions aubio_get_rubberband_opts(const char_t *mode);

//static void aubio_timestretch_warmup (aubio_timestretch_t * p);

aubio_timestretch_t *
new_aubio_timestretch (const char_t * mode, smpl_t stretchratio, uint_t hopsize,
    uint_t samplerate)
{
  aubio_timestretch_t *p = AUBIO_NEW (aubio_timestretch_t);
  p->hopsize = hopsize;
  p->pitchscale = 1.;

  if ((sint_t)hopsize <= 0) {
    AUBIO_ERR("timestretch: hopsize should be > 0, got %d\n", hopsize);
    goto beach;
  }

  if ((sint_t)samplerate <= 0) {
    AUBIO_ERR("timestretch: samplerate should be > 0, got %d\n", samplerate);
    goto beach;
  }

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

  p->rb = rubberband_new(samplerate, 1, p->rboptions, p->stretchratio, p->pitchscale);
  if (!p->rb) goto beach;

  p->samplerate = samplerate;

  //aubio_timestretch_warmup(p);

  return p;

beach:
  del_aubio_timestretch(p);
  return NULL;
}

#if 0
static void
aubio_timestretch_warmup (aubio_timestretch_t * p)
{
  // warm up rubber band
  //AUBIO_WRN("timestretch: warming-up\n");
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
  fvec_t *input = new_fvec(p->hopsize);
  while (aubio_timestretch_push(p, input, input->length) < (int)latency) {
    //sint_t available = aubio_timestretch_get_available(p);
    //AUBIO_WRN("timestretch: warmup got %d, latency: %d\n", available, latency);
  }
  del_fvec(input);
}
#endif

void
del_aubio_timestretch (aubio_timestretch_t * p)
{
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
  if (!p->rb) {
    AUBIO_ERR("timestretch: could not set stretch ratio,"
       " rubberband not created\n");
    return AUBIO_FAIL;
  }
  if (stretch >= MIN_STRETCH_RATIO && stretch <= MAX_STRETCH_RATIO) {
    p->stretchratio = stretch;
    rubberband_set_time_ratio(p->rb, 1./p->stretchratio);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("timestretch: could not set stretch ratio to '%f',"
        " should be in the range [%.2f, %.2f].\n", stretch,
        MIN_STRETCH_RATIO, MAX_STRETCH_RATIO);
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
  if (!p->rb) {
    AUBIO_ERR("timestretch: could not set pitch scale,"
       " rubberband not created\n");
    return AUBIO_FAIL;
  }
  if (pitchscale >= 0.0625  && pitchscale <= 4.) {
    p->pitchscale = pitchscale;
    rubberband_set_pitch_scale(p->rb, p->pitchscale);
    return AUBIO_OK;
  } else {
    AUBIO_ERR("timestretch: could not set pitchscale to '%f',"
        " should be in the range [0.0625, 4.].\n", pitchscale);
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
    AUBIO_ERR("timestretch: could not set transpose to '%f',"
        " should be in the range [-24; 24].\n", transpose);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_timestretch_get_transpose(aubio_timestretch_t * p)
{
  return 12. * LOG(p->pitchscale) / LOG(2.0);
}

sint_t
aubio_timestretch_push(aubio_timestretch_t *p, fvec_t *input, uint_t length)
{
  // push new samples to rubberband, return available
  int available;
  int eof = (input->length != length) ? 1 : 0;
  rubberband_process(p->rb, (const float* const*)&(input->data), length, eof);
  available = rubberband_available(p->rb);
  //AUBIO_WRN("timestretch: processed %d, %d available, eof: %d\n",
  //    length, available, eof);
  return available;
}

sint_t
aubio_timestretch_get_available(aubio_timestretch_t *p) {
  return rubberband_available(p->rb);
}

void
aubio_timestretch_do(aubio_timestretch_t * p, fvec_t * out, uint_t * read)
{
  // now retrieve the samples and write them into out->data
  int available = rubberband_available(p->rb);
  if (available >= (int)out->length) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), out->length);
    *read = out->length;
  } else if (available > 0) {
    // this occurs each time the end of file is reached
    //AUBIO_WRN("timestretch: short read\n");
    rubberband_retrieve(p->rb, (float* const*)&(out->data), available);
    fvec_t zeros; zeros.length = out->length - available; zeros.data = out->data + available;
    fvec_zeros(&zeros);
    *read = available;
  } else {
    // this may occur if the previous was a short read available == hopsize
    fvec_zeros(out);
    *read = 0;
  }
}

uint_t
aubio_timestretch_reset(aubio_timestretch_t *p)
{
  uint_t err = AUBIO_OK;
  if (p->rb) {
    rubberband_reset(p->rb);
  }
  return err;
}

#endif

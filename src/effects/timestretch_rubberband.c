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

  aubio_source_t *source;
  uint_t source_hopsize;          /**< hop size at which the source is read */
  fvec_t *in;
  uint_t eof;

  RubberBandState rb;
  RubberBandOptions rboptions;

#ifdef HAVE_THREADS
  pthread_t read_thread;
  pthread_mutex_t read_mutex;
  pthread_cond_t read_avail;
  pthread_cond_t read_request;
  sint_t available;
#endif
};

extern RubberBandOptions aubio_get_rubberband_opts(const char_t *mode);

static void aubio_timestretch_warmup (aubio_timestretch_t * p);
static sint_t aubio_timestretch_fetch(aubio_timestretch_t *p, uint_t fetch);
#ifdef HAVE_THREADS
static void *aubio_timestretch_readfn(void *p);
#endif

aubio_timestretch_t *
new_aubio_timestretch (const char_t * uri, const char_t * mode,
    smpl_t stretchratio, uint_t hopsize, uint_t samplerate)
{
  aubio_timestretch_t *p = AUBIO_NEW (aubio_timestretch_t);
  p->samplerate = samplerate;
  p->hopsize = hopsize;
  //p->source_hopsize = 2048;
  p->source_hopsize = hopsize;
  p->pitchscale = 1.;
  p->eof = 0;

  p->source = new_aubio_source(uri, samplerate, p->source_hopsize);
  if (!p->source) goto beach;
  if (samplerate == 0 ) p->samplerate = aubio_source_get_samplerate(p->source);

  p->in = new_fvec(p->source_hopsize);

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
  rubberband_set_max_process_size(p->rb, p->source_hopsize);
  //rubberband_set_debug_level(p->rb, 10);

#ifdef HAVE_THREADS
  pthread_mutex_init(&p->read_mutex, 0);
  pthread_cond_init (&p->read_avail, 0);
  pthread_cond_init (&p->read_request, 0);
  pthread_create(&p->read_thread, 0, aubio_timestretch_readfn, p);
  //AUBIO_DBG("timestretch: new_ waiting for warmup, got %d available\n", p->available);
  pthread_mutex_lock(&p->read_mutex);
  pthread_cond_wait(&p->read_avail, &p->read_mutex);
  pthread_mutex_unlock(&p->read_mutex);
  //AUBIO_DBG("timestretch: new_ warm up success, got %d available\n", p->available);
#else
  aubio_timestretch_warmup(p);
#endif

  return p;

beach:
  del_aubio_timestretch(p);
  return NULL;
}

#ifdef HAVE_THREADS
void *
aubio_timestretch_readfn(void *z)
{
  aubio_timestretch_t *p = z;
  // signal main-thread when we are done
  //AUBIO_WRN("timestretch: read_thread locking, got %d available\n", p->available);
  pthread_mutex_lock(&p->read_mutex);
  aubio_timestretch_warmup(p);
  //AUBIO_WRN("timestretch: signaling warmup\n");
  pthread_cond_signal(&p->read_avail);
  //AUBIO_WRN("timestretch: unlocking in readfn\n");
  pthread_mutex_unlock(&p->read_mutex);
  AUBIO_WRN("timestretch: entering readfn loop\n");
  while(1) { //p->available < (int)p->hopsize && p->eof != 1) {
    //AUBIO_WRN("timestretch: locking in readfn\n");
    pthread_mutex_lock(&p->read_mutex);
    p->available = aubio_timestretch_fetch(p, p->hopsize);
    //AUBIO_WRN("timestretch: read_thread read %d\n", p->available);
    // signal main-thread when we are done
    //AUBIO_WRN("timestretch: signaling new read\n");
    pthread_cond_signal(&p->read_avail);
    if (p->eof != 1) {
      pthread_cond_wait(&p->read_request, &p->read_mutex);
    }
    if (p->eof == 1) {
      AUBIO_WRN("timestretch: read_thread eof reached %d, %d/%d\n", p->available,
        p->hopsize, p->source_hopsize);
      pthread_mutex_unlock(&p->read_mutex);
      break;
    }
    //AUBIO_WRN("timestretch: unlocking in readfn\n");
    pthread_mutex_unlock(&p->read_mutex);
  }
#if 1
  pthread_mutex_lock(&p->read_mutex);
  //AUBIO_WRN("timestretch: signaling end\n");
  pthread_cond_signal(&p->read_avail);
  pthread_mutex_unlock(&p->read_mutex);
#endif
  //AUBIO_WRN("timestretch: exiting readfn\n");
  pthread_exit(NULL);
}
#endif

static void
aubio_timestretch_warmup (aubio_timestretch_t * p)
{
  // warm up rubber band
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
#ifdef HAVE_THREADS
  p->available = aubio_timestretch_fetch(p, latency);
#else
  aubio_timestretch_fetch(p, latency);
#endif
}

void
del_aubio_timestretch (aubio_timestretch_t * p)
{
#ifdef HAVE_THREADS
  pthread_mutex_lock(&p->read_mutex);
  pthread_cond_signal(&p->read_request);
  //pthread_cond_wait(&p->read_avail, &p->read_mutex);
  pthread_mutex_unlock(&p->read_mutex);
#if 1
  void *threadfn;
  if ((p->eof == 0) && (pthread_cancel(p->read_thread))) {
      AUBIO_WRN("timestretch: cancelling thread failed\n");
  }
  if (pthread_join(p->read_thread, &threadfn)) {
      AUBIO_WRN("timestretch: joining thread failed\n");
  }
#endif
  pthread_mutex_destroy(&p->read_mutex);
  pthread_cond_destroy(&p->read_avail);
  pthread_cond_destroy(&p->read_request);
#endif
  if (p->in) del_fvec(p->in);
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
    AUBIO_WRN("timestretch: could not set stretch ratio to %.2f\n", stretch);
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
    AUBIO_WRN("timestretch: could not set pitchscale to %.2f\n", pitchscale);
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
    AUBIO_WRN("timestretch: could not set transpose to %.2f\n", transpose);
    return AUBIO_FAIL;
  }
}

smpl_t
aubio_timestretch_get_transpose(aubio_timestretch_t * p)
{
  return 12. * LOG(p->pitchscale) / LOG(2.0);
}

sint_t
aubio_timestretch_fetch(aubio_timestretch_t *p, uint_t length)
{
  uint_t source_read = p->source_hopsize;
  // read more samples from source until we have enough available or eof is reached
  int available = rubberband_available(p->rb);
  while ((available < (int)length) && (p->eof == 0)) {
    aubio_source_do(p->source, p->in, &source_read);
    if (source_read < p->source_hopsize) {
      p->eof = 1;
    }
    rubberband_process(p->rb, (const float* const*)&(p->in->data), source_read, p->eof);
    available = rubberband_available(p->rb);
  }
  return available;
}

void
aubio_timestretch_do (aubio_timestretch_t * p, fvec_t * out, uint_t * read)
{
#ifndef HAVE_THREADS
  int available = aubio_timestretch_fetch(p, p->hopsize);
#else /* HAVE_THREADS */
  int available;
  pthread_mutex_lock(&p->read_mutex);
  if (p->eof != 1) {
    // signal a read request
    pthread_cond_signal(&p->read_request);
    // wait for an available signal
    pthread_cond_wait(&p->read_avail, &p->read_mutex);
  } else {
    available = rubberband_available(p->rb);
  }
#endif /* HAVE_THREADS */
  // now retrieve the samples and write them into out->data
  if (available >= (int)p->hopsize) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), p->hopsize);
    *read = p->hopsize;
  } else if (available > 0) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), available);
    *read = available;
  } else {
    fvec_zeros(out);
    *read = 0;
  }
#ifdef HAVE_THREADS
  pthread_mutex_unlock(&p->read_mutex);
#endif
}

uint_t
aubio_timestretch_seek (aubio_timestretch_t *p, uint_t pos)
{
  uint_t err = AUBIO_OK;
#if HAVE_THREADS
  AUBIO_WRN("timestretch: seek_ waiting for warmup, got %d available\n", p->available);
  pthread_mutex_lock(&p->read_mutex);
#endif
  p->eof = 0;
  rubberband_reset(p->rb);
  err = aubio_source_seek(p->source, pos);
#if HAVE_THREADS
  p->available = 0;
  void *threadfn;
  if ((p->eof == 0) && (pthread_cancel(p->read_thread) == 0)) {
      AUBIO_WRN("timestretch: cancelling thread failed\n");
  }
  if (pthread_join(p->read_thread, &threadfn)) {
      AUBIO_WRN("timestretch: joining thread failed\n");
  }
  pthread_create(&p->read_thread, 0, aubio_timestretch_readfn, p);
  pthread_cond_wait(&p->read_avail, &p->read_mutex);
  pthread_mutex_unlock(&p->read_mutex);
  //AUBIO_WRN("timestretch: seek_ warm up success, got %d available\n", p->available);
#endif
  return err;
}

#endif

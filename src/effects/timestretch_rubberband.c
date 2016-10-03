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

  uint_t opened;
  const char_t *uri;
#ifdef HAVE_THREADS
  pthread_t read_thread;
  pthread_mutex_t read_mutex;
  pthread_cond_t read_avail;
  pthread_cond_t read_request;
  pthread_t open_thread;
  pthread_mutex_t open_mutex;
  uint_t open_thread_running;
  sint_t available;
  uint_t started;
  uint_t finish;
#endif
};

extern RubberBandOptions aubio_get_rubberband_opts(const char_t *mode);

static void aubio_timestretch_warmup (aubio_timestretch_t * p);
static sint_t aubio_timestretch_fetch(aubio_timestretch_t *p, uint_t fetch);
#ifdef HAVE_THREADS
static void *aubio_timestretch_readfn(void *p);
static void *aubio_timestretch_openfn(void *z);
#endif

aubio_timestretch_t *
new_aubio_timestretch (const char_t * uri, const char_t * mode,
    smpl_t stretchratio, uint_t hopsize, uint_t samplerate)
{
  aubio_timestretch_t *p = AUBIO_NEW (aubio_timestretch_t);
  p->hopsize = hopsize;
  //p->source_hopsize = 2048;
  p->source_hopsize = hopsize;
  p->pitchscale = 1.;

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

  p->in = new_fvec(p->source_hopsize);

#ifndef HAVE_THREADS
  if (aubio_timestretch_queue(p, uri, samplerate)) goto beach;
  aubio_timestretch_warmup(p);
#else
  p->started = 0;
  p->finish = 0;
  p->open_thread_running = 0;
  //p->uri = uri;
  p->eof = 0;
  //p->samplerate = samplerate;
  //if (aubio_timestretch_open(p, uri, samplerate)) goto beach;
  pthread_mutex_init(&p->open_mutex, 0);
  pthread_mutex_init(&p->read_mutex, 0);
  pthread_cond_init (&p->read_avail, 0);
  pthread_cond_init (&p->read_request, 0);
  //AUBIO_WRN("timestretch: creating thread\n");
  pthread_create(&p->read_thread, 0, aubio_timestretch_readfn, p);
  //AUBIO_DBG("timestretch: new_ waiting for warmup, got %d available\n", p->available);
  pthread_mutex_lock(&p->read_mutex);
  aubio_timestretch_queue(p, uri, samplerate);
#if 0
  pthread_cond_wait(&p->read_avail, &p->read_mutex);
  if (!p->opened) {
    goto beach;
  }
#endif
  pthread_mutex_unlock(&p->read_mutex);
  //AUBIO_DBG("timestretch: new_ warm up success, got %d available\n", p->available);
#endif

  return p;

beach:
  del_aubio_timestretch(p);
  return NULL;
}

#define HAVE_OPENTHREAD 1
//#undef HAVE_OPENTHREAD

uint_t
aubio_timestretch_queue(aubio_timestretch_t *p, const char_t* uri, uint_t samplerate)
{
#ifdef HAVE_THREADS
#ifdef HAVE_OPENTHREAD
  if (p->open_thread_running) {
#if 1
    if (pthread_cancel(p->open_thread)) {
      AUBIO_WRN("timestretch: cancelling open thread failed\n");
      return AUBIO_FAIL;
    } else {
      AUBIO_WRN("timestretch: previous open of '%s' cancelled\n", p->uri);
    }
    p->open_thread_running = 0;
#else
    void *threadfn;
    if (pthread_join(p->open_thread, &threadfn)) {
      AUBIO_WRN("timestretch: failed joining existing open thread\n");
      return AUBIO_FAIL;
    }
#endif
  }
  //AUBIO_WRN("timestretch: queueing %s\n", uri);
  //pthread_mutex_lock(&p->read_mutex);
  p->opened = 0;
  p->started = 0;
  p->available = 0;
  p->uri = uri;
  p->samplerate = samplerate;
  //AUBIO_WRN("timestretch: creating thread\n");
  pthread_create(&p->open_thread, 0, aubio_timestretch_openfn, p);
#endif
  //pthread_mutex_unlock(&p->read_mutex);
  return AUBIO_OK;
}

uint_t
aubio_timestretch_open(aubio_timestretch_t *p, const char_t* uri, uint_t samplerate)
{
  uint_t err = AUBIO_FAIL;
  p->available = 0;
  pthread_mutex_lock(&p->open_mutex);
  p->open_thread_running = 1;
#else
  uint_t err = AUBIO_FAIL;
#endif
  p->opened = 0;
  if (p->source) del_aubio_source(p->source);
  p->source = new_aubio_source(uri, samplerate, p->source_hopsize);
  if (!p->source) goto fail;
  p->uri = uri;
  p->samplerate = aubio_source_get_samplerate(p->source);
  p->eof = 0;

  if (p->rb == NULL) {
    AUBIO_WRN("timestretch: creating with stretch: %.2f pitchscale: %.2f\n",
        p->stretchratio, p->pitchscale);
    p->rb = rubberband_new(p->samplerate, 1, p->rboptions, p->stretchratio, p->pitchscale);
    //rubberband_set_debug_level(p->rb, 10);
    rubberband_set_max_process_size(p->rb, p->source_hopsize);
  } else {
    if (samplerate != p->samplerate) {
      AUBIO_WRN("timestretch: samplerate change requested, but not implemented\n");
    }
    rubberband_reset(p->rb);
  }
  p->opened = 1;
  err = AUBIO_OK;
  goto unlock;
fail:
  p->opened = 2;
  AUBIO_ERR("timestretch: opening %s failed\n", uri);
unlock:
#ifdef HAVE_THREADS
  p->open_thread_running = 0;
  pthread_mutex_unlock(&p->open_mutex);
  //AUBIO_WRN("timestretch: failed opening %s at %dHz\n", uri, samplerate);
#endif
  return err;
}

#ifdef HAVE_THREADS
void *
aubio_timestretch_openfn(void *z) {
  aubio_timestretch_t *p = z;
  int oldtype;
  pthread_setcancelstate(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);
  //AUBIO_WRN("timestretch: creating thread\n");
  void *ret;
  uint_t err = aubio_timestretch_open(p, p->uri, p->samplerate);
  ret = &err;
  pthread_exit(ret);
}
#endif

uint_t
aubio_timestretch_get_opened(aubio_timestretch_t *p)
{
  if (p == NULL) return 0;
  else return p->opened;
}

#ifdef HAVE_THREADS
void *
aubio_timestretch_readfn(void *z)
{
  aubio_timestretch_t *p = z;
  //AUBIO_WRN("timestretch: entering thread with %s at %dHz\n", p->uri, p->samplerate);
  while(1) { //p->available < (int)p->hopsize && p->eof != 1) {
    //AUBIO_WRN("timestretch: locking in readfn\n");
    pthread_mutex_lock(&p->read_mutex);
#if 1
    if (p->opened == 2) {
      pthread_cond_signal(&p->read_avail);
    } else
    if (p->opened == 0) {
#ifdef HAVE_OPENTHREAD
      //(!aubio_timestretch_open(p, p->uri, p->samplerate)) {
      void * threadfn;
      if (p->open_thread_running && pthread_join(p->open_thread, &threadfn)) {
        AUBIO_WRN("timestretch: failed to join opening thread %s at %dHz in thread "
            "(opened: %d, playing: %d, eof: %d)\n",
            p->uri, p->samplerate, p->opened, p->started, p->eof);
      }
#else
      //AUBIO_WRN("timestretch: opening source %s\n", p->uri);
      if (!aubio_timestretch_open(p, p->uri, p->samplerate)) {
        AUBIO_WRN("timestretch: opened %s at %dHz in thread "
            "(opened: %d, playing: %d, eof: %d)\n",
            p->uri, p->samplerate, p->opened, p->started, p->eof);
        //pthread_cond_signal(&p->read_avail);
      } else {
        AUBIO_WRN("timestretch: failed opening %s, exiting thread\n", p->uri);
        //pthread_cond_signal(&p->read_avail);
        //pthread_mutex_unlock(&p->read_mutex);
        //goto end;
      }
#endif
    } else
    if (!p->started && !p->eof) {
#endif
      // fetch the first few samples and mark as started
      aubio_timestretch_warmup(p);
      pthread_cond_signal(&p->read_avail);
      //pthread_cond_wait(&p->read_request, &p->read_mutex);
      p->started = 1;
    } else if (!p->eof) {
      // fetch at least p->hopsize stretched samples
      p->available = aubio_timestretch_fetch(p, p->hopsize);
      // signal available frames
      pthread_cond_signal(&p->read_avail);
      if (p->eof != 1) {
        // the end of file was not reached yet, wait for the next read_request
        pthread_cond_wait(&p->read_request, &p->read_mutex);
      } else {
        // eof was reached, do not wait for a read request and mark as stopped
        p->started = 0;
      }
    } else {
      //pthread_cond_signal(&p->read_avail);
      pthread_cond_wait(&p->read_request, &p->read_mutex);
      //AUBIO_WRN("timestretch: finished idle in readfn\n");
      if (p->finish) pthread_exit(NULL);
    }
    //AUBIO_WRN("timestretch: unlocking in readfn\n");
    pthread_mutex_unlock(&p->read_mutex);
  }
end:
  //AUBIO_WRN("timestretch: exiting readfn\n");
  pthread_exit(NULL);
}
#endif

static void
aubio_timestretch_warmup (aubio_timestretch_t * p)
{
  // warm up rubber band
  //AUBIO_WRN("timestretch: warming-up\n");
  unsigned int latency = MAX(p->hopsize, rubberband_get_latency(p->rb));
#ifdef HAVE_THREADS
  p->available = aubio_timestretch_fetch(p, latency);
#else
  aubio_timestretch_fetch(p, latency);
#endif
  //AUBIO_WRN("timestretch: warmup got %d\n", latency);
}

void
del_aubio_timestretch (aubio_timestretch_t * p)
{
#ifdef HAVE_THREADS
  void *threadfn;
  //AUBIO_WRN("timestretch: entering delete\n");
  if (p->open_thread_running) {
    if (pthread_cancel(p->open_thread)) {
      AUBIO_WRN("timestretch: cancelling open thread failed\n");
    }
    if (pthread_join(p->open_thread, &threadfn)) {
      AUBIO_WRN("timestretch: joining open thread failed\n");
    }
  }
  if (!p->opened) goto cleanup;
  pthread_mutex_lock(&p->read_mutex);
  p->finish = 1;
  pthread_cond_signal(&p->read_request);
  //pthread_cond_wait(&p->read_avail, &p->read_mutex);
  pthread_mutex_unlock(&p->read_mutex);
  if ((p->eof == 0) && (pthread_cancel(p->read_thread))) {
    AUBIO_WRN("timestretch: cancelling thread failed\n");
  }
  if (pthread_join(p->read_thread, &threadfn)) {
    AUBIO_WRN("timestretch: joining thread failed\n");
  }
  pthread_mutex_destroy(&p->read_mutex);
  pthread_cond_destroy(&p->read_avail);
  pthread_cond_destroy(&p->read_request);
cleanup:
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
  if (!p->rb) {
    AUBIO_WRN("timestretch: could not set stretch ratio, rubberband not created\n");
    return AUBIO_FAIL;
  }
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
  if (!p->rb) {
    AUBIO_WRN("timestretch: could not set pitch scale, rubberband not created\n");
    return AUBIO_FAIL;
  }
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
  if (p->source == NULL) {
    AUBIO_ERR("timestretch: trying to fetch on NULL source\n");
    return 0;
  }
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
#if 1
  if (!p->opened) {
    // this may occur if _do was was called while being opened
    //AUBIO_WRN("timestretch: calling _do before opening a file\n");
    pthread_cond_signal(&p->read_request);
    //available = 0;
    //pthread_cond_wait(&p->read_avail, &p->read_mutex);
    available = 0; //p->available;
  } else
#endif
  if (p->eof != 1) {
    //AUBIO_WRN("timestretch: calling _do after opening a file\n");
    // signal a read request
    pthread_cond_signal(&p->read_request);
    // wait for an available signal
    pthread_cond_wait(&p->read_avail, &p->read_mutex);
    available = p->available;
  } else {
    available = rubberband_available(p->rb);
    //AUBIO_WRN("timestretch: reached eof (%d/%d)\n", p->hopsize, available);
  }
  pthread_mutex_unlock(&p->read_mutex);
#endif /* HAVE_THREADS */
  // now retrieve the samples and write them into out->data
  if (available >= (int)p->hopsize) {
    rubberband_retrieve(p->rb, (float* const*)&(out->data), p->hopsize);
    *read = p->hopsize;
  } else if (available > 0) {
    // this occurs each time the end of file is reached
    //AUBIO_WRN("timestretch: short read\n");
    rubberband_retrieve(p->rb, (float* const*)&(out->data), available);
    *read = available;
  } else {
    // this may occur if the previous was a short read available == hopsize
    fvec_zeros(out);
    *read = 0;
  }
#ifdef HAVE_THREADS
  //pthread_mutex_unlock(&p->read_mutex);
#endif
}

uint_t
aubio_timestretch_seek (aubio_timestretch_t *p, uint_t pos)
{
  uint_t err = AUBIO_OK;
#if HAVE_THREADS
  if (p == NULL) {
    AUBIO_WRN("seeking but object not set yet (ignoring)\n");
    return AUBIO_FAIL;
  }
  pthread_mutex_lock(&p->read_mutex);
  if (p->open_thread_running) {
    //AUBIO_WRN("seeking but opening thread not completed yet (ignoring)\n");
    err = AUBIO_OK;
    goto beach;
  }
  if (!p->opened || !p->source) {
    //AUBIO_WRN("timestretch: seeking but source not opened yet (ignoring)\n");
    err = AUBIO_OK;
    goto beach;
  }
#endif
  p->eof = 0;
  if (p->rb) {
    rubberband_reset(p->rb);
  }
#ifdef HAVE_THREADS
#ifdef HAVE_OPENTHREAD
  pthread_mutex_lock(&p->open_mutex);
#endif
#endif
  if (p->source) {
    err = aubio_source_seek(p->source, pos);
  } else {
    AUBIO_WRN("timestretch: seeking but p->source not created?!\n");
    err = AUBIO_FAIL;
    goto beach;
  }
#if HAVE_THREADS
  pthread_mutex_unlock(&p->open_mutex);
  p->available = 0;
  p->started = 1;
beach:
  pthread_mutex_unlock(&p->read_mutex);
#else
beach:
#endif
  return err;
}

#endif

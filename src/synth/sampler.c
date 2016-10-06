/*
  Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

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
#include "fmat.h"
#include "io/source.h"
#include "synth/sampler.h"

#define HAVE_THREADS 1
#define READER_THREAD_ON 0
#if 0
#undef HAVE_THREADS
#endif

#ifdef HAVE_THREADS
#include <pthread.h>
#endif

struct _aubio_sampler_t {
  uint_t samplerate;
  uint_t blocksize;
  aubio_source_t *source;
  const char_t *uri;
  uint_t playing;
  uint_t opened;
  uint_t loop;
  uint_t finished;              // end of file was reached
  uint_t eof;                   // end of file is now
#ifdef HAVE_THREADS
  // file reading thread
  pthread_t read_thread;
  uint_t threaded_read;         // use reading thread?
  pthread_mutex_t read_mutex;
  pthread_cond_t read_avail;
  pthread_cond_t read_request;
  pthread_t open_thread;        // file opening thread
  pthread_mutex_t open_mutex;
  uint_t waited;                // number of frames skipped while opening
  const char_t *next_uri;
  uint_t open_thread_running;
  sint_t available;             // number of samples currently available
  uint_t started;               // source warmed up
  uint_t read_thread_finish;    // flag to tell reading thread to exit
#endif
};

#ifdef HAVE_THREADS
static void *aubio_sampler_openfn(void *p);
static void *aubio_sampler_readfn(void *p);
static void aubio_sampler_open_opening_thread(aubio_sampler_t *o);
static void aubio_sampler_open_reading_thread(aubio_sampler_t *o);
static void aubio_sampler_close_opening_thread(aubio_sampler_t *o);
static void aubio_sampler_close_reading_thread(aubio_sampler_t *o);
#endif

aubio_sampler_t *new_aubio_sampler(uint_t blocksize, uint_t samplerate)
{
  aubio_sampler_t *s = AUBIO_NEW(aubio_sampler_t);
  if ((sint_t)blocksize < 1) {
    AUBIO_ERR("sampler: got blocksize %d, but can not be < 1\n", blocksize);
    goto beach;
  }
  s->samplerate = samplerate;
  s->blocksize = blocksize;
  s->source = NULL;
  s->playing = 0;
  s->loop = 0;
  s->uri = NULL;
  s->finished = 1;
  s->eof = 0;
  s->opened = 0;
  s->available = 0;

#ifdef HAVE_THREADS
  s->threaded_read = READER_THREAD_ON;
  aubio_sampler_open_opening_thread(s);
  if (s->threaded_read) {
    aubio_sampler_open_reading_thread(s);
  }
#endif

  return s;
beach:
  AUBIO_FREE(s);
  return NULL;
}

#ifdef HAVE_THREADS
void aubio_sampler_open_opening_thread(aubio_sampler_t *s) {
  pthread_mutex_init(&s->open_mutex, 0);
  s->waited = 0;
  s->open_thread = 0;
  s->open_thread_running = 0;
}

void aubio_sampler_open_reading_thread(aubio_sampler_t *s) {
  s->read_thread_finish = 0;
  pthread_mutex_init(&s->read_mutex, 0);
  pthread_cond_init (&s->read_avail, 0);
  pthread_cond_init (&s->read_request, 0);
  pthread_create(&s->read_thread, 0, aubio_sampler_readfn, s);
}

void aubio_sampler_close_opening_thread(aubio_sampler_t *o) {
  // clean up opening thread
  void *threadret;
  pthread_mutex_destroy(&o->open_mutex);
  if (o->open_thread_running) {
    if (pthread_cancel(o->open_thread)) {
      AUBIO_WRN("sampler: cancelling file opening thread failed\n");
    }
  }
  if (o->open_thread && pthread_join(o->open_thread, &threadret)) {
    AUBIO_WRN("sampler: joining file opening thread failed\n");
  }
  pthread_mutex_destroy(&o->open_mutex);
}

void aubio_sampler_close_reading_thread(aubio_sampler_t *o) {
  // clean up reading thread
  void *threadret;
  if (!o->read_thread) return;
  o->read_thread_finish = 1;
  pthread_cond_signal(&o->read_request);
  if (pthread_cancel(o->read_thread)) {
    AUBIO_WRN("sampler: cancelling file reading thread failed\n");
  }
  if (pthread_join(o->read_thread, &threadret)) {
    AUBIO_WRN("sampler: joining file reading thread failed\n");
  }
  pthread_mutex_destroy(&o->read_mutex);
  pthread_cond_destroy(&o->read_avail);
  pthread_cond_destroy(&o->read_request);
}
#endif

uint_t aubio_sampler_load( aubio_sampler_t * o, const char_t * uri )
{
  uint_t ret = AUBIO_FAIL;
  aubio_source_t *oldsource = o->source, *newsource = NULL;
  newsource = new_aubio_source(uri, o->samplerate, o->blocksize);
  if (newsource) {
    o->source = newsource;
    if (oldsource) del_aubio_source(oldsource);
    if (o->samplerate == 0) {
      o->samplerate = aubio_source_get_samplerate(o->source);
    }
    o->uri = uri;
    o->finished = 0;
    o->eof = 0;
    o->opened = 1;
    ret = AUBIO_OK;
    //AUBIO_WRN("sampler: loaded %s\n", o->uri);
  } else {
    o->source = NULL;
    if (oldsource) del_aubio_source(oldsource);
    o->playing = 0;
    o->uri = NULL;
    o->finished = 1;
    o->eof = 0;
    o->opened = 0;
    AUBIO_WRN("sampler: failed loading %s\n", uri);
  }
  return ret;
}

#ifdef HAVE_THREADS
static void *aubio_sampler_openfn(void *z) {
  aubio_sampler_t *p = z;
  uint_t err;
  int oldtype;
  void *ret;
  pthread_setcancelstate(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);
  pthread_mutex_lock(&p->open_mutex);
  p->open_thread_running = 1;
  err = aubio_sampler_load(p, p->next_uri);
  p->open_thread_running = 0;
  pthread_mutex_unlock(&p->open_mutex);
  ret = &err;
  pthread_exit(ret);
}
#endif

uint_t
aubio_sampler_queue(aubio_sampler_t *o, const char_t *uri)
{
#ifdef HAVE_THREADS
  uint_t ret = AUBIO_OK;
  /* open uri in open_thread */
  if (o->open_thread_running) {
    // cancel previous open_thread
    if (pthread_cancel(o->open_thread)) {
      AUBIO_WRN("sampler: cancelling open thread failed\n");
      return AUBIO_FAIL;
    } else {
      AUBIO_WRN("sampler: previous open of %s cancelled while opening %s\n",
          o->next_uri, uri);
    }
    o->open_thread_running = 0;
  }
  void *threadret;
  if (o->open_thread && pthread_join(o->open_thread, &threadret)) {
    AUBIO_WRN("sampler: joining thread failed\n");
  }
  if (pthread_mutex_trylock(&o->open_mutex)) {
    AUBIO_WRN("sampler: failed locking in queue\n");
    ret = AUBIO_FAIL;
    goto lock_failed;
  }
  o->opened = 0; // while opening
  o->started = 0;
  o->available = 0;
  o->next_uri = uri;
  o->waited = 0;
  if (pthread_create(&o->open_thread, 0, aubio_sampler_openfn, o) != 0) {
    AUBIO_ERR("sampler: failed creating opening thread\n");
    ret = AUBIO_FAIL;
    goto thread_create_failed;
  }

thread_create_failed:
  pthread_mutex_unlock(&o->open_mutex);
lock_failed:
  if (ret == AUBIO_OK) {
    //AUBIO_WRN("sampler: queued %s\n", uri);
  } else {
    AUBIO_ERR("sampler: queueing %s failed\n", uri);
  }
  return ret;
#else
  AUBIO_WRN("sampler: opening %s, not queueing (not compiled with threading)\n", uri);
  return aubio_sampler_load(o, uri);
#endif
}

#ifdef HAVE_THREADS
void *aubio_sampler_readfn(void *z) {
  aubio_sampler_t *p = z;
  while(1) {
    pthread_mutex_lock(&p->read_mutex);
    if (1) {
      // idle
      pthread_cond_wait(&p->read_request, &p->read_mutex);
      if (p->read_thread_finish) {
        goto done;
      }
    }
    pthread_mutex_unlock(&p->read_mutex);
  }
done:
  //AUBIO_WRN("sampler: exiting reading thread\n");
  pthread_mutex_unlock(&p->read_mutex);
  pthread_exit(NULL);
}
#endif

void
aubio_sampler_fetch_from_source(aubio_sampler_t *o, fvec_t *output, uint_t *read) {
  if (o->opened == 1 && o->source && !o->finished)
    aubio_source_do(o->source, output, read);
}

void
aubio_sampler_fetch_from_source_multi(aubio_sampler_t *o, fmat_t *output, uint_t *read) {
  aubio_source_do_multi(o->source, output, read);
}

#if 0
void
aubio_sampler_fetch_from_array(aubio_sampler_t *o, fvec_t *output, uint_t *read) {
  // TODO
}
#endif

uint_t
aubio_sampler_get_samplerate (aubio_sampler_t *o)
{
  return o->samplerate;
}

uint_t
aubio_sampler_get_opened (aubio_sampler_t *o)
{
  return o->opened; //== 1 ? AUBIO_OK : AUBIO_FAIL;
}

uint_t
aubio_sampler_get_finished(aubio_sampler_t *o)
{
  return o->finished;
}

uint_t
aubio_sampler_get_eof (aubio_sampler_t *o)
{
  return o->eof;
}

uint_t
aubio_sampler_get_waited_opening (aubio_sampler_t *o, uint_t waited) {
#ifdef HAVE_THREADS
  if (o->playing) {
    if (!o->opened) {
      o->waited += waited;
    } else if (o->waited) {
      //AUBIO_WRN("sampler: waited %d frames (%.2fms) while opening %s\n",
      //    o->waited, 1000.*o->waited/(smpl_t)o->samplerate, o->uri);
      uint_t waited = o->waited;
      o->waited = 0;
      return waited;
    }
  }
#endif
  return 0;
}

uint_t
aubio_sampler_seek(aubio_sampler_t * o, uint_t pos)
{
  uint_t ret = AUBIO_FAIL;
  o->finished = 0;
  if (!o->opened) return AUBIO_OK;
  if (o->source) {
    ret = aubio_source_seek(o->source, pos);
  }
  return ret;
}

void
aubio_sampler_do_eof (aubio_sampler_t * o)
{
  o->finished = 1;
  o->eof = 1;
  if (!o->loop) {
    o->playing = 0;
  } else {
    aubio_sampler_seek(o, 0);
  }
}

void aubio_sampler_do ( aubio_sampler_t * o, fvec_t * output, uint_t *read)
{
  o->eof = 0;
  if (o->opened == 1 && o->playing) {
    aubio_sampler_fetch_from_source(o, output, read);
    if (*read < o->blocksize) {
      aubio_sampler_do_eof (o);
      if (*read > 0) {
        // TODO pull (hopsize - read) frames
        //memset(...  tail , 0)
      }
    }
  } else {
    fvec_zeros(output);
    *read = 0; //output->length;
  }
}

void aubio_sampler_do_multi ( aubio_sampler_t * o, fmat_t * output, uint_t *read)
{
  o->eof = 0;
  if (o->playing) {
    aubio_sampler_fetch_from_source_multi(o, output, read);
    if (*read < o->blocksize) {
      aubio_sampler_do_eof (o);
      if (*read > 0) {
        // TODO pull (hopsize - read) frames
        //memset(...  tail , 0)
      }
    }
  } else {
    fmat_zeros(output);
    *read = 0;
  }
}

uint_t aubio_sampler_get_playing ( const aubio_sampler_t * o )
{
  return o->playing;
}

uint_t aubio_sampler_set_playing ( aubio_sampler_t * o, uint_t playing )
{
  o->playing = (playing == 1) ? 1 : 0;
  return 0;
}

uint_t aubio_sampler_get_loop ( aubio_sampler_t * o )
{
  return o->loop;
}

uint_t aubio_sampler_set_loop ( aubio_sampler_t * o, uint_t loop )
{
  o->loop = (loop == 1) ? 1 : 0;
  return 0;
}

uint_t aubio_sampler_play ( aubio_sampler_t * o )
{
  //aubio_source_seek (o->source, 0);
  return aubio_sampler_set_playing (o, 1);
}

uint_t aubio_sampler_stop ( aubio_sampler_t * o )
{
  return aubio_sampler_set_playing (o, 0);
}

uint_t aubio_sampler_loop ( aubio_sampler_t * o )
{
  aubio_sampler_set_loop(o, 1);
  aubio_sampler_seek(o, 0);
  return aubio_sampler_set_playing (o, 1);
}

uint_t aubio_sampler_trigger ( aubio_sampler_t * o )
{
  aubio_sampler_set_loop(o, 0);
  aubio_sampler_seek(o, 0);
  return aubio_sampler_set_playing (o, 1);
}

void del_aubio_sampler( aubio_sampler_t * o )
{
#ifdef HAVE_THREADS
  // close opening thread
  aubio_sampler_close_opening_thread(o);
  // close reading thread
  aubio_sampler_close_reading_thread(o);
#endif
  if (o->source) {
    del_aubio_source(o->source);
  }
  AUBIO_FREE(o);
}

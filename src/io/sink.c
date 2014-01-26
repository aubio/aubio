/*
  Copyright (C) 2012 Paul Brossier <piem@aubio.org>

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
#include "aubio_priv.h"
#include "fvec.h"
#include "io/sink.h"
#ifdef __APPLE__
#include "io/sink_apple_audio.h"
#endif /* __APPLE__ */
#ifdef HAVE_SNDFILE
#include "io/sink_sndfile.h"
#endif
#ifdef HAVE_WAVWRITE
#include "io/sink_wavwrite.h"
#endif

typedef void (*aubio_sink_do_t)(aubio_sink_t * s, fvec_t * data, uint_t write);
#if 0
typedef void (*aubio_sink_do_multi_t)(aubio_sink_t * s, fmat_t * data, uint_t * read);
typedef uint_t (*aubio_sink_get_samplerate_t)(aubio_sink_t * s);
typedef uint_t (*aubio_sink_get_channels_t)(aubio_sink_t * s);
#endif
typedef void (*del_aubio_sink_t)(aubio_sink_t * s);

struct _aubio_sink_t { 
  void *sink;
  aubio_sink_do_t s_do;
#if 0
  aubio_sink_do_multi_t s_do_multi;
  aubio_sink_get_samplerate_t s_get_samplerate;
  aubio_sink_get_channels_t s_get_channels;
#endif
  del_aubio_sink_t s_del;
};

aubio_sink_t * new_aubio_sink(char_t * uri, uint_t samplerate) {
  aubio_sink_t * s = AUBIO_NEW(aubio_sink_t);
#ifdef __APPLE__
  s->sink = (void *)new_aubio_sink_apple_audio(uri, samplerate);
  if (s->sink) {
    s->s_do = (aubio_sink_do_t)(aubio_sink_apple_audio_do);
    s->s_del = (del_aubio_sink_t)(del_aubio_sink_apple_audio);
    return s;
  }
#endif /* __APPLE__ */
#if HAVE_SNDFILE
  s->sink = (void *)new_aubio_sink_sndfile(uri, samplerate);
  if (s->sink) {
    s->s_do = (aubio_sink_do_t)(aubio_sink_sndfile_do);
    s->s_del = (del_aubio_sink_t)(del_aubio_sink_sndfile);
    return s;
  }
#endif /* HAVE_SNDFILE */
#if HAVE_WAVWRITE
  s->sink = (void *)new_aubio_sink_wavwrite(uri, samplerate);
  if (s->sink) {
    s->s_do = (aubio_sink_do_t)(aubio_sink_wavwrite_do);
    s->s_del = (del_aubio_sink_t)(del_aubio_sink_wavwrite);
    return s;
  }
#endif /* HAVE_WAVWRITE */
  AUBIO_ERROR("sink: failed creating aubio sink with %s\n", uri);
  AUBIO_FREE(s);
  return NULL;
}

void aubio_sink_do(aubio_sink_t * s, fvec_t * write_data, uint_t write) {
  s->s_do((void *)s->sink, write_data, write);
}

void del_aubio_sink(aubio_sink_t * s) {
  if (!s) return;
  s->s_del((void *)s->sink);
  AUBIO_FREE(s);
  return;
}

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

struct _aubio_sink_t { 
  void *sink;
};

aubio_sink_t * new_aubio_sink(char_t * uri, uint_t samplerate) {
  aubio_sink_t * s = AUBIO_NEW(aubio_sink_t);
#ifdef __APPLE__
  s->sink = (void *)new_aubio_sink_apple_audio(uri, samplerate);
  if (s->sink) return s;
#else /* __APPLE__ */
#if HAVE_SNDFILE
  s->sink = (void *)new_aubio_sink_sndfile(uri, samplerate);
  if (s->sink) return s;
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
  AUBIO_ERROR("failed creating aubio sink with %s\n", uri);
  AUBIO_FREE(s);
  return NULL;
}

void aubio_sink_do(aubio_sink_t * s, fvec_t * write_data, uint_t write) {
#ifdef __APPLE__
  aubio_sink_apple_audio_do((aubio_sink_apple_audio_t *)s->sink, write_data, write);
#else /* __APPLE__ */
#if HAVE_SNDFILE
  aubio_sink_sndfile_do((aubio_sink_sndfile_t *)s->sink, write_data, write);
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
}

void del_aubio_sink(aubio_sink_t * s) {
  if (!s) return;
#ifdef __APPLE__
  del_aubio_sink_apple_audio((aubio_sink_apple_audio_t *)s->sink);
#else /* __APPLE__ */
#if HAVE_SNDFILE
  del_aubio_sink_sndfile((aubio_sink_sndfile_t *)s->sink);
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
  AUBIO_FREE(s);
  return;
}

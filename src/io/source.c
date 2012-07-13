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
#include "io/source.h"
#ifdef __APPLE__
#include "io/source_apple_audio_file.h"
#endif /* __APPLE__ */
#ifdef HAVE_SNDFILE
#include "io/source_sndfile.h"
#endif

struct _aubio_source_t { 
  void *source;
};

aubio_source_t * new_aubio_source(char_t * uri, uint_t samplerate, uint_t hop_size) {
  aubio_source_t * s = AUBIO_NEW(aubio_source_t);
#ifdef __APPLE__
  s->source= (void *)new_aubio_source_apple_audio(uri, samplerate, hop_size);
  if (s->source) return s;
#else /* __APPLE__ */
#if HAVE_SNDFILE
  s->source= (void *)new_aubio_source_sndfile(uri, samplerate, hop_size);
  if (s->source) return s;
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
  if (s->source == NULL) return NULL;
}

void aubio_source_do(aubio_source_t * s, fvec_t * data, uint_t * read) {
#ifdef __APPLE__
  aubio_source_apple_audio_do((aubio_source_apple_audio_t *)s->source, data, read);
#else /* __APPLE__ */
#if HAVE_SNDFILE
  aubio_source_sndfile_do((aubio_source_sndfile_t *)s->source, data, read);
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
}

void del_aubio_source(aubio_source_t * s) {
#ifdef __APPLE__
  del_aubio_source_apple_audio((aubio_source_apple_audio_t *)s->source);
#else /* __APPLE__ */
#if HAVE_SNDFILE
  del_aubio_source_sndfile((aubio_source_sndfile_t *)s->source);
#endif /* HAVE_SNDFILE */
#endif /* __APPLE__ */
  AUBIO_FREE(s);
}


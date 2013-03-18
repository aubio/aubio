/*
  Copyright (C) 2013 Paul Brossier <piem@aubio.org>

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

#ifndef _AUBIO_IO_AUDIO_UNIT_H
#define _AUBIO_IO_AUDIO_UNIT_H

/** \file

*/

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _aubio_audio_unit_t aubio_audio_unit_t;

typedef uint_t (*aubio_audio_unit_callback_t) (void * closure, float *ibuf, float *obuf, uint_t size);


aubio_audio_unit_t * new_aubio_audio_unit(uint_t samplerate, uint_t inchannels,
    uint_t outchannels, uint_t blocksize);
uint_t del_aubio_audio_unit(aubio_audio_unit_t *o);

#ifdef __cplusplus
}
#endif

#endif /* _AUBIO_IO_AUDIO_UNIT_H */

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

#ifndef _AUBIO_SOURCE_H
#define _AUBIO_SOURCE_H

#ifdef __cplusplus
extern "C" {
#endif

/** \file

  Media source 

*/

typedef struct _aubio_source_t aubio_source_t;
aubio_source_t * new_aubio_source(char_t * uri, uint_t samplerate, uint_t hop_size);
void aubio_source_do(aubio_source_t * s, fvec_t * read_data, uint_t * read);
void del_aubio_source(aubio_source_t * s);

#ifdef __cplusplus
}
#endif

#endif /* _AUBIO_SOURCE_H */

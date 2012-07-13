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

struct _aubio_sink_t { 
  uint_t hopsize;
  uint_t samplerate;
};

aubio_sink_t * new_aubio_sink(char_t * uri, uint_t hop_size, uint_t samplerate) {
  aubio_sink_t * s = AUBIO_NEW(aubio_sink_t);
  return s;
}

void aubio_sink_do(aubio_sink_t * s, fvec_t * write_data, uint_t * written) {
}

void del_aubio_sink(aubio_sink_t * s) {
  AUBIO_FREE(s);
  return;
}

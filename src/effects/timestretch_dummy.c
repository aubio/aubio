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

#ifndef HAVE_RUBBERBAND

#include "fvec.h"
#include "effects/timestretch.h"

// TODO fallback time stretching implementation

struct _aubio_timestretch_t
{
  void *dummy;
};

void
aubio_timestretch_do (aubio_timestretch_t * o UNUSED, fvec_t * out UNUSED,
    uint_t * read UNUSED)
{
}

void del_aubio_timestretch (aubio_timestretch_t * o UNUSED) {
}

aubio_timestretch_t *
new_aubio_timestretch (const char_t * method UNUSED,
    smpl_t pitchscale UNUSED, uint_t hop_size UNUSED, uint_t samplerate UNUSED)
{
  AUBIO_ERR ("timestretch: aubio was not compiled with rubberband\n");
  return NULL;
}

uint_t aubio_timestretch_set_stretch (aubio_timestretch_t * o UNUSED, smpl_t stretch UNUSED)
{
  return AUBIO_FAIL;
}

smpl_t aubio_timestretch_get_stretch (aubio_timestretch_t * o UNUSED)
{
  return 1.;
}

uint_t aubio_timestretch_set_pitchscale (aubio_timestretch_t * o UNUSED, smpl_t pitchscale UNUSED)
{
  return AUBIO_FAIL;
}

uint_t aubio_timestretch_get_samplerate (aubio_timestretch_t * o UNUSED) {
  return 0;
}

smpl_t aubio_timestretch_get_pitchscale (aubio_timestretch_t * o UNUSED)
{
  return 1.;
}

uint_t aubio_timestretch_set_transpose (aubio_timestretch_t * o UNUSED, smpl_t transpose UNUSED) {
  return AUBIO_FAIL;
}

smpl_t aubio_timestretch_get_transpose (aubio_timestretch_t * o UNUSED) {
  return 0.;
}

uint_t aubio_timestretch_get_latency (aubio_timestretch_t * o UNUSED) {
  return 0.;
}

uint_t aubio_timestretch_reset(aubio_timestretch_t *o UNUSED) {
  return AUBIO_FAIL;
}

sint_t aubio_timestretch_push(aubio_timestretch_t * o UNUSED, fvec_t * in
    UNUSED, uint_t length UNUSED) {
  return AUBIO_FAIL;
}

sint_t aubio_timestretch_get_available(aubio_timestretch_t * o UNUSED) {
  return AUBIO_FAIL;
}
// end of dummy implementation

#endif /* HAVE_RUBBERBAND */

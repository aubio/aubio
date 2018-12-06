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
#include "effects/pitchshift.h"

// TODO fallback pitch shifting implementation

struct _aubio_pitchshift_t
{
  void *dummy;
};

void aubio_pitchshift_do (aubio_pitchshift_t * o UNUSED, const fvec_t * in UNUSED,
    fvec_t * out UNUSED) {
}

void del_aubio_pitchshift (aubio_pitchshift_t * o UNUSED) {
}

aubio_pitchshift_t *new_aubio_pitchshift (const char_t * method UNUSED,
    smpl_t pitchscale UNUSED, uint_t hop_size UNUSED, uint_t samplerate UNUSED)
{
  AUBIO_ERR ("aubio was not compiled with rubberband\n");
  return NULL;
}

uint_t aubio_pitchshift_set_pitchscale (aubio_pitchshift_t * o UNUSED, smpl_t pitchscale UNUSED)
{
  return AUBIO_FAIL;
}

smpl_t aubio_pitchshift_get_pitchscale (aubio_pitchshift_t * o UNUSED)
{
  return 1.;
}

uint_t aubio_pitchshift_set_transpose (aubio_pitchshift_t * o UNUSED, smpl_t transpose UNUSED) {
  return AUBIO_FAIL;
}

smpl_t aubio_pitchshift_get_transpose (aubio_pitchshift_t * o UNUSED) {
  return 0.;
}

uint_t aubio_pitchshift_get_latency (aubio_pitchshift_t * o UNUSED) {
  return 0.;
}

// end of dummy implementation

#endif /* HAVE_RUBBERBAND */

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

uint_t
aubio_io_validate_samplerate(const char_t *kind, const char_t *path, uint_t samplerate)
{
  if ((sint_t)(samplerate) <= 0) {
    AUBIO_ERR("%s: failed creating %s, samplerate should be positive, not %d\n",
        kind, path, samplerate);
    return AUBIO_FAIL;
  }
  if ((sint_t)samplerate > AUBIO_MAX_SAMPLERATE) {
    AUBIO_ERR("%s: failed creating %s, samplerate %dHz is too large\n",
        kind, path, samplerate);
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

uint_t
aubio_io_validate_channels(const char_t *kind, const char_t *path, uint_t channels)
{
  if ((sint_t)(channels) <= 0) {
    AUBIO_ERR("sink_%s: failed creating %s, channels should be positive, not %d\n",
        kind, path, channels);
    return AUBIO_FAIL;
  }
  if (channels > AUBIO_MAX_CHANNELS) {
    AUBIO_ERR("sink_%s: failed creating %s, too many channels (%d but %d available)\n",
        kind, path, channels, AUBIO_MAX_CHANNELS);
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

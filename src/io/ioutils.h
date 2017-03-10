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

#ifndef AUBIO_IOUTILS_H
#define AUBIO_IOUTILS_H

/** \file

  Simple utility functions to validate input parameters.

*/

#ifdef __cplusplus
extern "C" {
#endif

/** validate samplerate

  \param kind       the object kind to report on
  \param path       the object properties to report on
  \param samplerate the object properties to report on
  \return  0 if ok, non-zero if validation failed

 */
uint_t aubio_io_validate_samplerate(const char_t *kind, const char_t *path,
    uint_t samplerate);

/** validate number of channels

  \param kind       the object kind to report on
  \param path       the object properties to report on
  \param channels   the object properties to report on
  \return  0 if ok, non-zero if validation failed

 */
uint_t aubio_io_validate_channels(const char_t *kind, const char_t *path,
    uint_t channels);

#ifdef __cplusplus
}
#endif

#endif /* AUBIO_IOUTILS_H */

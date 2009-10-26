/*
  Copyright (C) 2007-2009 Paul Brossier <piem@aubio.org>

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

/** @file
 * compute spectrum centroid of a cvec object
 */

#ifndef _SPECTRAL_CENTROID_H
#define _SPECTRAL_CENTROID_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * spectrum centroid computed on a cvec
 */
smpl_t aubio_spectral_centroid(cvec_t * input, smpl_t samplerate);

#ifdef __cplusplus
}
#endif

#endif /* _SPECTRAL_CENTROID_H */

/*
  Copyright (C) 2018 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_WIN_HACKS
#define strncasecmp _strnicmp
#endif

const char_t *aubio_str_get_extension(const char_t *filename)
{
  // find last occurence of dot character
  const char_t *ext;
  if (!filename) return NULL;
  ext = strrchr(filename, '.');
  if (!ext || ext == filename) return "";
  else return ext + 1;
}

uint_t aubio_str_extension_matches(const char_t *ext, const char_t *pattern)
{
  return ext && pattern && (strncasecmp(ext, pattern, PATH_MAX) == 0);
}

uint_t aubio_str_path_has_extension(const char_t *filename,
    const char_t *pattern)
{
  const char_t *ext = aubio_str_get_extension(filename);
  return aubio_str_extension_matches(ext, pattern);
}

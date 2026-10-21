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

#define _GNU_SOURCE
#include "aubio_priv.h"

#ifdef HAVE_WIN_HACKS
#define strncasecmp _strnicmp
#else
#include <strings.h>
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

char_t *aubio_str_copy_path(const char_t *path)
{
  size_t len;
  char_t *copy;
  if (!path) return NULL;
  // a longer path can't be opened, and copying part of it names another file
  len = strnlen(path, PATH_MAX);
  if (len == PATH_MAX) {
    AUBIO_ERR("path is %d characters or longer: %.64s...\n", PATH_MAX, path);
    return NULL;
  }
  copy = AUBIO_ARRAY(char_t, len + 1);
  if (copy) AUBIO_MEMCPY(copy, path, len + 1);
  return copy;
}

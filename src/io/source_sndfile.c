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

#ifdef HAVE_SNDFILE

#include <sndfile.h>

#include "aubio_priv.h"
#include "source_sndfile.h"
#include "fvec.h"

#define MAX_CHANNELS 6
#define MAX_SIZE 4096

struct _aubio_source_sndfile_t {
  uint_t hop_size;
  uint_t samplerate;
  uint_t channels;
  int input_samplerate;
  int input_channels;
  int input_format;
  char_t *path;
  SNDFILE *handle;
  uint_t scratch_size;
  smpl_t *scratch_data;
};

aubio_source_sndfile_t * new_aubio_source_sndfile(char_t * path, uint_t samplerate, uint_t hop_size) {
  aubio_source_sndfile_t * s = AUBIO_NEW(aubio_source_sndfile_t);

  if (path == NULL) {
    AUBIO_ERR("Aborted opening null path\n");
    return NULL;
  }

  s->hop_size = hop_size;
  s->samplerate = samplerate;
  s->channels = 1;
  s->path = path;

  // try opening the file, geting the info in sfinfo
  SF_INFO sfinfo;
  AUBIO_MEMSET(&sfinfo, 0, sizeof (sfinfo));
  s->handle = sf_open (s->path, SFM_READ, &sfinfo);

  if (s->handle == NULL) {
    /* show libsndfile err msg */
    AUBIO_ERR("Failed opening %s: %s\n", s->path, sf_strerror (NULL));
    return NULL;
  }	

  if (sfinfo.channels > MAX_CHANNELS) { 
    AUBIO_ERR("Not able to process more than %d channels\n", MAX_CHANNELS);
    return NULL;
  }

  /* get input specs */
  s->input_samplerate = sfinfo.samplerate;
  s->input_channels   = sfinfo.channels;
  s->input_format     = sfinfo.format;

  if (s->samplerate != s->input_samplerate) {
    AUBIO_ERR("resampling not implemented yet\n");
    return NULL;
  }
  
  s->scratch_size = s->hop_size*s->input_channels;
  /* allocate data for de/interleaving reallocated when needed. */
  if (s->scratch_size >= MAX_SIZE * MAX_CHANNELS) {
    AUBIO_ERR("%d x %d exceeds maximum aubio_source_sndfile buffer size %d\n",
        s->hop_size, s->input_channels, MAX_CHANNELS * MAX_CHANNELS);
    return NULL;
  }
  s->scratch_data = AUBIO_ARRAY(float,s->scratch_size);

  return s;
}

void aubio_source_sndfile_do(aubio_source_sndfile_t * s, fvec_t * read_data, uint_t * read){
  sf_count_t read_frames;
  int i,j, input_channels = s->input_channels;
  int aread;
  /* do actual reading */
  read_frames = sf_read_float (s->handle, s->scratch_data, s->scratch_size);

  aread = (int)FLOOR(read_frames/(float)input_channels);

  /* de-interleaving and down-mixing data  */
  for (j = 0; j < aread; j++) {
    read_data->data[j] = 0;
    for (i = 0; i < input_channels; i++) {
      read_data->data[j] += (smpl_t)s->scratch_data[input_channels*j+i];
    }
    read_data->data[j] /= (smpl_t)input_channels;
  }
  *read = aread;
}

void del_aubio_source_sndfile(aubio_source_sndfile_t * s){
  if (sf_close(s->handle)) {
    AUBIO_ERR("Error closing file %s: %s", s->path, sf_strerror (NULL));
  }
  AUBIO_FREE(s->scratch_data);
  AUBIO_FREE(s);
}

#endif /* HAVE_SNDFILE */

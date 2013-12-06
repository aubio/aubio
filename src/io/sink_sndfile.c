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
#include "sink_sndfile.h"
#include "fvec.h"

#define MAX_CHANNELS 6
#define MAX_SIZE 4096

struct _aubio_sink_sndfile_t {
  uint_t samplerate;
  uint_t channels;
  char_t *path;

  uint_t max_size;

  SNDFILE *handle;
  uint_t scratch_size;
  smpl_t *scratch_data;
};

aubio_sink_sndfile_t * new_aubio_sink_sndfile(char_t * path, uint_t samplerate) {
  aubio_sink_sndfile_t * s = AUBIO_NEW(aubio_sink_sndfile_t);

  if (path == NULL) {
    AUBIO_ERR("Aborted opening null path\n");
    return NULL;
  }

  s->samplerate = samplerate;
  s->max_size = MAX_SIZE;
  s->channels = 1;
  s->path = path;

  /* set output format */
  SF_INFO sfinfo;
  AUBIO_MEMSET(&sfinfo, 0, sizeof (sfinfo));
  sfinfo.samplerate = s->samplerate;
  sfinfo.channels   = s->channels;
  sfinfo.format     = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  /* try creating the file */
  s->handle = sf_open (s->path, SFM_WRITE, &sfinfo);

  if (s->handle == NULL) {
    /* show libsndfile err msg */
    AUBIO_ERR("Failed opening %s. %s\n", s->path, sf_strerror (NULL));
    AUBIO_FREE(s);
    return NULL;
  }	

  s->scratch_size = s->max_size*s->channels;
  /* allocate data for de/interleaving reallocated when needed. */
  if (s->scratch_size >= MAX_SIZE * MAX_CHANNELS) {
    AUBIO_ERR("%d x %d exceeds maximum aubio_sink_sndfile buffer size %d\n",
        s->max_size, s->channels, MAX_CHANNELS * MAX_CHANNELS);
    AUBIO_FREE(s);
    return NULL;
  }
  s->scratch_data = AUBIO_ARRAY(float,s->scratch_size);

  return s;
}

void aubio_sink_sndfile_do(aubio_sink_sndfile_t *s, fvec_t * write_data, uint_t write){
  uint_t i, j,	channels = s->channels;
  int nsamples = channels*write;
  smpl_t *pwrite;

  if (write > s->max_size) {
    AUBIO_WRN("trying to write %d frames, but only %d can be written at a time",
      write, s->max_size);
    write = s->max_size;
  }

  /* interleaving data  */
  for ( i = 0; i < channels; i++) {
    pwrite = (smpl_t *)write_data->data;
    for (j = 0; j < write; j++) {
      s->scratch_data[channels*j+i] = pwrite[j];
    }
  }

  sf_count_t written_frames = sf_write_float (s->handle, s->scratch_data, nsamples);
  if (written_frames/channels != write) {
    AUBIO_WRN("trying to write %d frames to %s, but only %d could be written",
      write, s->path, (uint_t)written_frames);
  }
  return;
}

void del_aubio_sink_sndfile(aubio_sink_sndfile_t * s){
  if (!s) return;
  if (sf_close(s->handle)) {
    AUBIO_ERR("Error closing file %s: %s", s->path, sf_strerror (NULL));
  }
  AUBIO_FREE(s->scratch_data);
  AUBIO_FREE(s);
}

#endif /* HAVE_SNDFILE */

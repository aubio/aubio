/*
  Copyright (C) 2014 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_WAVWRITE

#include "aubio_priv.h"
#include "sink_wavwrite.h"
#include "fvec.h"

#include <errno.h>

#define MAX_CHANNELS 6
#define MAX_SIZE 4096

#define FLOAT_TO_SHORT(x) (short)(x * 32768)

// swap endian of a short
#define SWAPS(x) ((x & 0xff) << 8) | ((x & 0xff00) >> 8)

// swap host to little endian
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define HTOLES(x) SWAPS(x)
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define HTOLES(x) x
#else
#define HTOLES(x) SWAPS(htons(x))
#endif

struct _aubio_sink_wavwrite_t {
  char_t *path;
  uint_t samplerate;
  uint_t channels;
  uint_t bitspersample;
  uint_t total_frames_written;

  FILE *fid;

  uint_t max_size;

  uint_t scratch_size;
  unsigned short *scratch_data;
};

unsigned char *write_little_endian (unsigned int s, unsigned char *str, unsigned int length);
unsigned char *write_little_endian (unsigned int s, unsigned char *str, unsigned int length) {
  uint_t i;
  for (i = 0; i < length; i++) {
    str[i] = s >> (i * 8);
  }
  return str;
}

aubio_sink_wavwrite_t * new_aubio_sink_wavwrite(char_t * path, uint_t samplerate) {
  aubio_sink_wavwrite_t * s = AUBIO_NEW(aubio_sink_wavwrite_t);
  unsigned char buf[5];
  uint_t byterate, blockalign;

  if (path == NULL) {
    AUBIO_ERR("sink_wavwrite: Aborted opening null path\n");
    goto beach;
  }
  if ((sint_t)samplerate < 0) {
    AUBIO_ERR("sink_wavwrite: Can not create %s with samplerate %d\n", path, samplerate);
    goto beach;
  }

  s->path = path;
  s->samplerate = samplerate;
  s->max_size = MAX_SIZE;
  s->channels = 1;
  s->bitspersample = 16;
  s->total_frames_written = 0;

  /* set output format */
  s->fid = fopen((const char *)path, "wb");
  if (!s->fid) {
    AUBIO_ERR("sink_wavwrite: could not open %s (%s)\n", s->path, strerror(errno));
    goto beach;
  }

  // ChunkID
  fwrite("RIFF", 4, 1, s->fid);

  // ChunkSize (0 for now, actual size will be written in _close)
  fwrite(write_little_endian(0, buf, 4), 4, 1, s->fid);

  // Format
  fwrite("WAVE", 4, 1, s->fid);

  // Subchunk1ID
  fwrite("fmt ", 4, 1, s->fid);

  // Subchunk1Size
  fwrite(write_little_endian(16, buf, 4), 4, 1, s->fid);

  // AudioFormat
  fwrite(write_little_endian(1, buf, 2), 2, 1, s->fid);

  // NumChannels
  fwrite(write_little_endian(s->channels, buf, 2), 2, 1, s->fid);

  // SampleRate
  fwrite(write_little_endian(s->samplerate, buf, 4), 4, 1, s->fid);

  // ByteRate
  byterate = s->samplerate * s->channels * s->bitspersample / 8;
  fwrite(write_little_endian(byterate, buf, 4), 4, 1, s->fid);

  // BlockAlign
  blockalign = s->channels * s->bitspersample / 8;
  fwrite(write_little_endian(blockalign, buf, 2), 2, 1, s->fid);

  // BitsPerSample
  fwrite(write_little_endian(s->bitspersample, buf, 2), 2, 1, s->fid);

  // Subchunk2ID
  fwrite("data", 4, 1, s->fid);

  // Subchunk1Size (0 for now, actual size will be written in _close)
  fwrite(write_little_endian(0, buf, 4), 4, 1, s->fid);

  s->scratch_size = s->max_size * s->channels;
  /* allocate data for de/interleaving reallocated when needed. */
  if (s->scratch_size >= MAX_SIZE * MAX_CHANNELS) {
    AUBIO_ERR("sink_wavwrite: %d x %d exceeds maximum buffer size %d\n",
        s->max_size, s->channels, MAX_CHANNELS * MAX_CHANNELS);
    AUBIO_FREE(s);
    return NULL;
  }
  s->scratch_data = AUBIO_ARRAY(unsigned short,s->scratch_size);

  return s;

beach:
  AUBIO_ERR("sink_wavwrite: can not write %s at samplerate %dHz\n",
      s->path, s->samplerate);
  del_aubio_sink_wavwrite(s);
  return NULL;
}


void aubio_sink_wavwrite_do(aubio_sink_wavwrite_t *s, fvec_t * write_data, uint_t write){
  uint_t i = 0, written_frames = 0;

  if (write > s->max_size) {
    AUBIO_WRN("sink_wavwrite: trying to write %d frames to %s, "
        "but only %d can be written at a time\n", write, s->path, s->max_size);
    write = s->max_size;
  }

  for (i = 0; i < write; i++) {
    s->scratch_data[i] = HTOLES(FLOAT_TO_SHORT(write_data->data[i]));
  }
  written_frames = fwrite(s->scratch_data, 2, write, s->fid);

  if (written_frames != write) {
    AUBIO_WRN("sink_wavwrite: trying to write %d frames to %s, "
        "but only %d could be written\n", write, s->path, written_frames);
  }
  s->total_frames_written += written_frames;
  return;
}

void aubio_sink_wavwrite_close(aubio_sink_wavwrite_t * s) {
  uint_t data_size = s->total_frames_written * s->bitspersample * s->channels / 8;
  unsigned char buf[5];
  if (!s->fid) return;
  // ChunkSize
  fseek(s->fid, 4, SEEK_SET);
  fwrite(write_little_endian(data_size + 36, buf, 4), 4, 1, s->fid);
  // Subchunk2Size
  fseek(s->fid, 40, SEEK_SET);
  fwrite(write_little_endian(data_size, buf, 4), 4, 1, s->fid);
  // close file
  if (fclose(s->fid)) {
    AUBIO_ERR("sink_wavwrite: Error closing file %s (%s)\n", s->path, strerror(errno));
  }
  s->fid = NULL;
}

void del_aubio_sink_wavwrite(aubio_sink_wavwrite_t * s){
  if (!s) return;
  aubio_sink_wavwrite_close(s);
  AUBIO_FREE(s->scratch_data);
  AUBIO_FREE(s);
}

#endif /* HAVE_WAVWRITE */

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

/*
  This file is largely inspired by `examples/encoder_example.c` in the
  libvorbis source package (versions 1.3.5 and later) available online at
  https://xiph.org/vorbis/
*/

#include "aubio_priv.h"

#ifdef HAVE_VORBISENC

#include "fmat.h"
#include "io/ioutils.h"

#include <vorbis/vorbisenc.h>
#include <string.h> // strerror
#include <errno.h> // errno
#include <time.h> // time

#define MAX_SIZE 4096

struct _aubio_sink_vorbis_t {
  FILE *fid;            // file id
  ogg_stream_state os;  // stream
  ogg_page og;          // page
  ogg_packet op;        // data packet
  vorbis_info vi;       // vorbis bitstream settings
  vorbis_comment vc;    // user comment
  vorbis_dsp_state vd;  // working state
  vorbis_block vb;      // working space

  uint_t samplerate;
  uint_t channels;
  char_t *path;
};

typedef struct _aubio_sink_vorbis_t aubio_sink_vorbis_t;

uint_t aubio_sink_vorbis_preset_channels(aubio_sink_vorbis_t *s,
    uint_t channels);
uint_t aubio_sink_vorbis_preset_samplerate(aubio_sink_vorbis_t *s,
    uint_t samplerate);
uint_t aubio_sink_vorbis_open(aubio_sink_vorbis_t *s);
uint_t aubio_sink_vorbis_close (aubio_sink_vorbis_t *s);
void del_aubio_sink_vorbis (aubio_sink_vorbis_t *s);

static uint_t aubio_sink_vorbis_write_page(aubio_sink_vorbis_t *s);

aubio_sink_vorbis_t * new_aubio_sink_vorbis (const char_t *uri,
    uint_t samplerate)
{
  aubio_sink_vorbis_t * s = AUBIO_NEW(aubio_sink_vorbis_t);

  if (!uri) {
    AUBIO_ERROR("sink_vorbis: Aborted opening null path\n");
    goto failure;
  }

  s->path = AUBIO_ARRAY(char_t, strnlen(uri, PATH_MAX) + 1);
  strncpy(s->path, uri, strnlen(uri, PATH_MAX) + 1);
  s->path[strnlen(uri, PATH_MAX)] = '\0';

  s->channels = 0;

  if ((sint_t)samplerate == 0)
    return s;

  aubio_sink_vorbis_preset_samplerate(s, samplerate);
  s->channels = 1;

  if (aubio_sink_vorbis_open(s) != AUBIO_OK)
    goto failure;

  return s;

failure:
  del_aubio_sink_vorbis(s);
  return NULL;
}

void del_aubio_sink_vorbis (aubio_sink_vorbis_t *s)
{
  if (s->fid) aubio_sink_vorbis_close(s);
  // clean up
  ogg_stream_clear(&s->os);
  vorbis_block_clear(&s->vb);
  vorbis_dsp_clear(&s->vd);
  vorbis_comment_clear(&s->vc);
  vorbis_info_clear(&s->vi);

  if (s->path) AUBIO_FREE(s->path);
  AUBIO_FREE(s);
}

uint_t aubio_sink_vorbis_open(aubio_sink_vorbis_t *s)
{
  float quality_mode = .9;

  if (s->samplerate == 0 || s->channels == 0) return AUBIO_FAIL;

  s->fid = fopen((const char *)s->path, "wb");
  if (!s->fid) {
    AUBIO_STRERR("sink_vorbis: Error opening file \'%s\' (%s)\n",
        s->path, errorstr);
    return AUBIO_FAIL;
  }

  vorbis_info_init(&s->vi);
  if (vorbis_encode_init_vbr(&s->vi, s->channels, s->samplerate, quality_mode))
  {
    AUBIO_ERR("sink_vorbis: vorbis_encode_init_vbr failed\n");
    return AUBIO_FAIL;
  }

  // add comment
  vorbis_comment_init(&s->vc);
  vorbis_comment_add_tag(&s->vc, "ENCODER", "aubio");

  // initalise analysis and block
  vorbis_analysis_init(&s->vd, &s->vi);
  vorbis_block_init(&s->vd, &s->vb);

  // pick randome serial number
  srand(time(NULL));
  ogg_stream_init(&s->os, rand());

  // write header
  {
    ogg_packet header;
    ogg_packet header_comm;
    ogg_packet header_code;

    vorbis_analysis_headerout(&s->vd, &s->vc, &header, &header_comm,
        &header_code);

    ogg_stream_packetin(&s->os, &header);
    ogg_stream_packetin(&s->os, &header_comm);
    ogg_stream_packetin(&s->os, &header_code);

    // make sure audio data will start on a new page
    while (1)
    {
      if (!ogg_stream_flush(&s->os, &s->og)) break;
      if (aubio_sink_vorbis_write_page(s)) return AUBIO_FAIL;
    }
  }

  return AUBIO_OK;
}

uint_t aubio_sink_vorbis_preset_samplerate(aubio_sink_vorbis_t *s,
    uint_t samplerate)
{
  if (aubio_io_validate_samplerate("sink_vorbis", s->path, samplerate))
    return AUBIO_FAIL;
  s->samplerate = samplerate;
  if (/* s->samplerate != 0 && */ s->channels != 0)
    return aubio_sink_vorbis_open(s);
  return AUBIO_OK;
}

uint_t aubio_sink_vorbis_preset_channels(aubio_sink_vorbis_t *s,
    uint_t channels)
{
  if (aubio_io_validate_channels("sink_vorbis", s->path, channels)) {
    return AUBIO_FAIL;
  }
  s->channels = channels;
  // automatically open when both samplerate and channels have been set
  if (s->samplerate != 0 /* && s->channels != 0 */) {
    return aubio_sink_vorbis_open(s);
  }
  return AUBIO_OK;
}

uint_t aubio_sink_vorbis_get_samplerate(const aubio_sink_vorbis_t *s)
{
  return s->samplerate;
}

uint_t aubio_sink_vorbis_get_channels(const aubio_sink_vorbis_t *s)
{
  return s->channels;
}

static
uint_t aubio_sink_vorbis_write_page(aubio_sink_vorbis_t *s) {
  int result;
  size_t wrote;
  wrote = fwrite(s->og.header, 1, s->og.header_len, s->fid);
  result = (wrote == (unsigned)s->og.header_len);
  wrote = fwrite(s->og.body, 1, s->og.body_len,     s->fid);
  result &= (wrote == (unsigned)s->og.body_len);
  if (result == 0) {
    AUBIO_STRERR("sink_vorbis: failed writing \'%s\' to disk (%s)\n",
        s->path, errorstr);
    return AUBIO_FAIL;
  }
  return AUBIO_OK;
}

static
void aubio_sink_vorbis_write(aubio_sink_vorbis_t *s)
{
  // pre-analysis
  while (vorbis_analysis_blockout(&s->vd, &s->vb) == 1) {

    vorbis_analysis(&s->vb, NULL);
    vorbis_bitrate_addblock(&s->vb);

    while (vorbis_bitrate_flushpacket(&s->vd, &s->op))
    {
      ogg_stream_packetin(&s->os, &s->op);

      while (1) {
        if (!ogg_stream_pageout (&s->os, &s->og)) break;
        aubio_sink_vorbis_write_page(s);
        if (ogg_page_eos(&s->og)) break;
      }
    }
  }
}

void aubio_sink_vorbis_do(aubio_sink_vorbis_t *s, fvec_t *write_data,
    uint_t write)
{
  uint_t c, v;
  uint_t length = aubio_sink_validate_input_length("sink_vorbis", s->path,
      MAX_SIZE, write_data->length, write);
  float **buffer = vorbis_analysis_buffer(&s->vd, (long)length);
  // fill buffer
  if (!write) {
    return;
  } else if (!buffer) {
    AUBIO_WRN("sink_vorbis: failed fetching buffer of size %d\n", write);
    return;
  } else {
    for (c = 0; c < s->channels; c++) {
      for (v = 0; v < length; v++) {
        buffer[c][v] = write_data->data[v];
      }
    }
    // tell vorbis how many frames were written
    vorbis_analysis_wrote(&s->vd, (long)length);
  }
  // write to file
  aubio_sink_vorbis_write(s);
}

void aubio_sink_vorbis_do_multi(aubio_sink_vorbis_t *s, fmat_t *write_data,
    uint_t write)
{
  uint_t c, v;
  uint_t channels = aubio_sink_validate_input_channels("sink_vorbis", s->path,
      s->channels, write_data->height);
  uint_t length = aubio_sink_validate_input_length("sink_vorbis", s->path,
      MAX_SIZE, write_data->length, write);
  float **buffer = vorbis_analysis_buffer(&s->vd, (long)length);
  // fill buffer
  if (!write) {
    return;
  } else if (!buffer) {
    AUBIO_WRN("sink_vorbis: failed fetching buffer of size %d\n", write);
    return;
  } else {
    for (c = 0; c < channels; c++) {
      for (v = 0; v < length; v++) {
        buffer[c][v] = write_data->data[c][v];
      }
    }
    // tell vorbis how many frames were written
    vorbis_analysis_wrote(&s->vd, (long)length);
  }

  aubio_sink_vorbis_write(s);
}

uint_t aubio_sink_vorbis_close (aubio_sink_vorbis_t *s)
{
  if (!s->fid) return AUBIO_FAIL;
  //mark the end of stream
  vorbis_analysis_wrote(&s->vd, 0);

  aubio_sink_vorbis_write(s);

  if (s->fid && fclose(s->fid)) {
    AUBIO_STRERR("sink_vorbis: Error closing file \'%s\' (%s)\n",
        s->path, errorstr);
    return AUBIO_FAIL;
  }
  s->fid = NULL;
  return AUBIO_OK;
}

#endif /* HAVE_VORBISENC */

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
  This file is largely inspired by `examples/c/encode/file/main.c` in the
  flac source package (versions 1.3.2 and later) available online at
  https://xiph.org/flac/
*/

#include "aubio_priv.h"

#ifdef HAVE_FLAC

#include "fmat.h"
#include "io/ioutils.h"

#include <FLAC/metadata.h>
#include <FLAC/stream_encoder.h>

#define MAX_WRITE_SIZE 4096

// swap host to little endian
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define HTOLES(x) SWAPS(x)
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define HTOLES(x) x
#else
#ifdef HAVE_WIN_HACKS
#define HTOLES(x) x
#else
#define HTOLES(x) SWAPS(htons(x))
#endif
#endif

// convert to short, taking care of endianness
#define FLOAT_TO_SHORT(x) (HTOLES((FLAC__int32)(x * 32768)))

struct _aubio_sink_flac_t {
  uint_t samplerate;
  uint_t channels;
  char_t *path;

  FILE *fid;            // file id
  FLAC__StreamEncoder* encoder;
  FLAC__int32 *buffer;
  FLAC__StreamMetadata **metadata;
};

typedef struct _aubio_sink_flac_t aubio_sink_flac_t;

uint_t aubio_sink_flac_preset_channels(aubio_sink_flac_t *s,
    uint_t channels);
uint_t aubio_sink_flac_preset_samplerate(aubio_sink_flac_t *s,
    uint_t samplerate);
uint_t aubio_sink_flac_open(aubio_sink_flac_t *s);
uint_t aubio_sink_flac_close (aubio_sink_flac_t *s);
void del_aubio_sink_flac (aubio_sink_flac_t *s);

#if 0
static void aubio_sink_flac_callback(const FLAC__StreamEncoder* encoder,
    FLAC__uint64 bytes_written, FLAC__uint64 samples_written,
    unsigned frames_writtten, unsigned total_frames_estimate,
    void *client_data);
#endif

aubio_sink_flac_t * new_aubio_sink_flac (const char_t *uri,
    uint_t samplerate)
{
  aubio_sink_flac_t * s = AUBIO_NEW(aubio_sink_flac_t);

  if (!uri) {
    AUBIO_ERROR("sink_flac: Aborted opening null path\n");
    goto failure;
  }

  s->path = AUBIO_ARRAY(char_t, strnlen(uri, PATH_MAX) + 1);
  strncpy(s->path, uri, strnlen(uri, PATH_MAX) + 1);
  s->path[strnlen(uri, PATH_MAX)] = '\0';

  s->channels = 0;
  s->samplerate = 0;

  if ((sint_t)samplerate == 0)
    return s;

  aubio_sink_flac_preset_samplerate(s, samplerate);
  s->channels = 1;

  if (aubio_sink_flac_open(s) != AUBIO_OK)
    goto failure;

  return s;

failure:
  del_aubio_sink_flac(s);
  return NULL;
}

void del_aubio_sink_flac (aubio_sink_flac_t *s)
{
  if (s->fid)
    aubio_sink_flac_close(s);
  if (s->buffer)
    AUBIO_FREE(s->buffer);
  if (s->path)
    AUBIO_FREE(s->path);
  AUBIO_FREE(s);
}

uint_t aubio_sink_flac_open(aubio_sink_flac_t *s)
{
  uint_t ret = AUBIO_FAIL;
  FLAC__bool ok = true;
  FLAC__StreamEncoderInitStatus init_status;
  FLAC__StreamMetadata_VorbisComment_Entry entry;
  const unsigned comp_level = 5;
  const unsigned bps = 16;

  if (s->samplerate == 0 || s->channels == 0) return AUBIO_FAIL;

  s->buffer = AUBIO_ARRAY(FLAC__int32, s->channels * MAX_WRITE_SIZE);
  if (!s->buffer) {
    AUBIO_ERR("sink_flac: failed allocating buffer for %s\n", s->path);
    return AUBIO_FAIL;
  }

  s->fid = fopen((const char *)s->path, "wb");
  if (!s->fid) {
    AUBIO_STRERR("sink_flac: Failed opening %s (%s)\n", s->path, errorstr);
    return AUBIO_FAIL;
  }

  if((s->encoder = FLAC__stream_encoder_new()) == NULL) {
    AUBIO_ERR("sink_flac: failed allocating encoder for %s\n", s->path);
    goto failure;
  }
  ok &= FLAC__stream_encoder_set_verify(s->encoder, true);
  ok &= FLAC__stream_encoder_set_compression_level(s->encoder, comp_level);
  ok &= FLAC__stream_encoder_set_channels(s->encoder, s->channels);
  ok &= FLAC__stream_encoder_set_bits_per_sample(s->encoder, bps);
  ok &= FLAC__stream_encoder_set_sample_rate(s->encoder, s->samplerate);
  // the total number of samples can not be estimated (streaming)
  // it will be set by the encoder in FLAC__stream_encoder_finish
  //ok &= FLAC__stream_encoder_set_total_samples_estimate(s->encoder, 0);

  if (!ok) {
    AUBIO_ERR("sink_flac: failed setting metadata for %s\n", s->path);
    goto failure;
  }

  s->metadata = AUBIO_ARRAY(FLAC__StreamMetadata*, 2);
  if (!s->metadata) {
    AUBIO_ERR("sink_flac: failed allocating memory for %s\n", s->path);
    goto failure;
  }

  s->metadata[0] = FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT);
  if (!s->metadata[0]) {
    AUBIO_ERR("sink_flac: failed allocating vorbis comment %s\n", s->path);
    goto failure;
  }

  s->metadata[1] = FLAC__metadata_object_new(FLAC__METADATA_TYPE_PADDING);
  if (!s->metadata[1]) {
    AUBIO_ERR("sink_flac: failed allocating vorbis comment %s\n", s->path);
    goto failure;
  }

  ok = FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry,
      "encoder", "aubio");
  ok &= FLAC__metadata_object_vorbiscomment_append_comment(s->metadata[0],
      entry, false);
  if (!ok) {
    AUBIO_ERR("sink_flac: failed setting metadata for %s\n", s->path);
    goto failure;
  }

  // padding length
  s->metadata[1]->length = 1234;
  if (!FLAC__stream_encoder_set_metadata(s->encoder, s->metadata, 2)) {
    AUBIO_ERR("sink_flac: failed setting metadata for %s\n", s->path);
    goto failure;
  }

  // initialize encoder
  init_status = FLAC__stream_encoder_init_file(s->encoder, s->path,
      NULL, NULL);
      //aubio_sink_flac_callback, s);
  if (init_status == FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_SAMPLE_RATE) {
    AUBIO_ERR("sink_flac: failed initilizing encoder for %s"
       " (invalid samplerate %d)\n", s->path, s->samplerate);
    goto failure;
  } else if (init_status ==
      FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_NUMBER_OF_CHANNELS) {
    AUBIO_ERR("sink_flac: failed initilizing encoder for %s"
       " (invalid number of channel %d)\n", s->path, s->channels);
    goto failure;
  } else if (init_status != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
    AUBIO_ERR("sink_flac: failed initilizing encoder for %s (%d)\n",
        s->path, (int)init_status);
    goto failure;
  }

  // mark success
  ret = AUBIO_OK;

failure:

  return ret;
}

uint_t aubio_sink_flac_preset_samplerate(aubio_sink_flac_t *s,
    uint_t samplerate)
{
  if (aubio_io_validate_samplerate("sink_flac", s->path, samplerate))
    return AUBIO_FAIL;
  s->samplerate = samplerate;
  if (s->samplerate != 0 && s->channels != 0)
    return aubio_sink_flac_open(s);
  return AUBIO_OK;
}

uint_t aubio_sink_flac_preset_channels(aubio_sink_flac_t *s,
    uint_t channels)
{
  if (aubio_io_validate_channels("sink_flac", s->path, channels)) {
    return AUBIO_FAIL;
  }
  s->channels = channels;
  // automatically open when both samplerate and channels have been set
  if (s->samplerate != 0 && s->channels != 0) {
    return aubio_sink_flac_open(s);
  }
  return AUBIO_OK;
}

uint_t aubio_sink_flac_get_samplerate(const aubio_sink_flac_t *s)
{
  return s->samplerate;
}

uint_t aubio_sink_flac_get_channels(const aubio_sink_flac_t *s)
{
  return s->channels;
}

static void aubio_sink_write_frames(aubio_sink_flac_t *s, uint_t length)
{
  // send to encoder
  if (!FLAC__stream_encoder_process_interleaved(s->encoder,
        (const FLAC__int32*)s->buffer, length)) {
    FLAC__StreamEncoderState state =
      FLAC__stream_encoder_get_state(s->encoder);
    AUBIO_WRN("sink_flac: error writing to %s (%s)\n",
        s->path, FLAC__StreamEncoderStateString[state]);
  }
}

void aubio_sink_flac_do(aubio_sink_flac_t *s, fvec_t *write_data,
    uint_t write)
{
  uint_t c, v;
  uint_t length = aubio_sink_validate_input_length("sink_flac", s->path,
      MAX_WRITE_SIZE, write_data->length, write);
  // fill buffer
  if (!write) {
    return;
  } else {
    for (c = 0; c < s->channels; c++) {
      for (v = 0; v < length; v++) {
        s->buffer[v * s->channels + c] = FLOAT_TO_SHORT(write_data->data[v]);
      }
    }
  }
  // send to encoder
  aubio_sink_write_frames(s, length);
}

void aubio_sink_flac_do_multi(aubio_sink_flac_t *s, fmat_t *write_data,
    uint_t write)
{
  uint_t c, v;
  uint_t channels = aubio_sink_validate_input_channels("sink_flac", s->path,
      s->channels, write_data->height);
  uint_t length = aubio_sink_validate_input_length("sink_flac", s->path,
      MAX_WRITE_SIZE, write_data->length, write);
  // fill buffer
  if (!write) {
    return;
  } else {
    for (c = 0; c < channels; c++) {
      for (v = 0; v < length; v++) {
        s->buffer[v * s->channels + c] = FLOAT_TO_SHORT(write_data->data[c][v]);
      }
    }
  }
  // send to encoder
  aubio_sink_write_frames(s, length);
}

uint_t aubio_sink_flac_close (aubio_sink_flac_t *s)
{
  uint_t ret = AUBIO_OK;

  if (!s->fid) return AUBIO_FAIL;

  if (s->encoder) {
    // mark the end of stream
    if (!FLAC__stream_encoder_finish(s->encoder)) {
      FLAC__StreamEncoderState state =
        FLAC__stream_encoder_get_state(s->encoder);
      AUBIO_ERR("sink_flac: Error closing encoder for %s (%s)\n",
          s->path, FLAC__StreamEncoderStateString[state]);
      ret &= AUBIO_FAIL;
    }

    FLAC__stream_encoder_delete(s->encoder);
  }

  if (s->metadata) {
    // clean up metadata after stream finished
    if (s->metadata[0])
      FLAC__metadata_object_delete(s->metadata[0]);
    if (s->metadata[1])
      FLAC__metadata_object_delete(s->metadata[1]);
    AUBIO_FREE(s->metadata);
  }

  if (s->fid && fclose(s->fid)) {
    AUBIO_STRERR("sink_flac: Error closing file %s (%s)\n", s->path, errorstr);
    ret &= AUBIO_FAIL;
  }
  s->fid = NULL;

  return ret;
}

#if 0
static void aubio_sink_flac_callback(const FLAC__StreamEncoder* encoder UNUSED,
    FLAC__uint64 bytes_written, FLAC__uint64 samples_written,
    unsigned frames_written, unsigned total_frames_estimate,
    void *client_data UNUSED)
{
  AUBIO_WRN("sink_flac: %d bytes_written, %d samples_written,"
      " %d/%d frames writen\n",
      bytes_written, samples_written, frames_written, total_frames_estimate);
}
#endif

#endif /* HAVE_FLAC */

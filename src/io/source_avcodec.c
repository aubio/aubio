/*
  Copyright (C) 2013 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_LIBAV

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavresample/avresample.h>
#include <libavutil/opt.h>
#include <stdlib.h>

#include "aubio_priv.h"
#include "fvec.h"
#include "fmat.h"
#include "source_avcodec.h"

#define AUBIO_AVCODEC_MAX_BUFFER_SIZE FF_MIN_BUFFER_SIZE

struct _aubio_source_avcodec_t {
  uint_t hop_size;
  uint_t samplerate;
  uint_t channels;

  // some data about the file
  char_t *path;
  uint_t input_samplerate;
  uint_t input_channels;

  // avcodec stuff
  AVFormatContext *avFormatCtx;
  AVCodecContext *avCodecCtx;
  AVFrame *avFrame;
  AVAudioResampleContext *avr;
  float *output;
  uint_t read_samples;
  uint_t read_index;
  sint_t selected_stream;
  uint_t eof;
  uint_t multi;
};

// hack to create or re-create the context the first time _do or _do_multi is called
void aubio_source_avcodec_reset_resampler(aubio_source_avcodec_t * s, uint_t multi);
void aubio_source_avcodec_readframe(aubio_source_avcodec_t *s, uint_t * read_samples);

aubio_source_avcodec_t * new_aubio_source_avcodec(char_t * path, uint_t samplerate, uint_t hop_size) {
  aubio_source_avcodec_t * s = AUBIO_NEW(aubio_source_avcodec_t);
  int err;
  if (path == NULL) {
    AUBIO_ERR("source_avcodec: Aborted opening null path\n");
    goto beach;
  }
  if ((sint_t)samplerate < 0) {
    AUBIO_ERR("source_avcodec: Can not open %s with samplerate %d\n", path, samplerate);
    goto beach;
  }
  if ((sint_t)hop_size <= 0) {
    AUBIO_ERR("source_avcodec: Can not open %s with hop_size %d\n", path, hop_size);
    goto beach;
  }

  s->hop_size = hop_size;
  s->channels = 1;
  s->path = path;

  // register all formats and codecs
  av_register_all();

  // if path[0] != '/'
  //avformat_network_init();

  // try opening the file and get some info about it
  AVFormatContext *avFormatCtx = s->avFormatCtx;
  avFormatCtx = NULL;
  if ( (err = avformat_open_input(&avFormatCtx, s->path, NULL, NULL) ) < 0 ) {
    char errorstr[256];
    av_strerror (err, errorstr, sizeof(errorstr));
    AUBIO_ERR("source_avcodec: Failed opening %s (%s)\n", s->path, errorstr);
    goto beach;
  }

  // try to make sure max_analyze_duration is big enough for most songs
  avFormatCtx->max_analyze_duration *= 100;

  // retrieve stream information
  if ( (err = avformat_find_stream_info(avFormatCtx, NULL)) < 0 ) {
    char errorstr[256];
    av_strerror (err, errorstr, sizeof(errorstr));
    AUBIO_ERR("source_avcodec: Could not find stream information " "for %s (%s)\n", s->path,
        errorstr);
    goto beach;
  }

  // dump information about file onto standard error
  //av_dump_format(avFormatCtx, 0, s->path, 0);

  // look for the first audio stream
  uint_t i;
  sint_t selected_stream = -1;
  for (i = 0; i < avFormatCtx->nb_streams; i++) {
    if (avFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
      if (selected_stream == -1) {
        selected_stream = i;
      } else {
        AUBIO_WRN("source_avcodec: More than one audio stream in %s, "
            "taking the first one\n", s->path);
      }
    }
  }
  if (selected_stream == -1) {
    AUBIO_ERR("source_avcodec: No audio stream in %s\n", s->path);
    goto beach;
  }
  //AUBIO_DBG("Taking stream %d in file %s\n", selected_stream, s->path);
  s->selected_stream = selected_stream;

  AVCodecContext *avCodecCtx = s->avCodecCtx;
  avCodecCtx = avFormatCtx->streams[selected_stream]->codec;
  AVCodec *codec = avcodec_find_decoder(avCodecCtx->codec_id);
  if (codec == NULL) {
    AUBIO_ERR("source_avcodec: Could not find decoder for %s", s->path);
    goto beach;
  }

  if ( ( err = avcodec_open2(avCodecCtx, codec, NULL) ) < 0) {
    char errorstr[256];
    av_strerror (err, errorstr, sizeof(errorstr));
    AUBIO_ERR("source_avcodec: Could not load codec for %s (%s)\n", s->path, errorstr);
    goto beach;
  }

  /* get input specs */
  s->input_samplerate = avCodecCtx->sample_rate;
  s->input_channels   = avCodecCtx->channels;
  //AUBIO_DBG("input_samplerate: %d\n", s->input_samplerate);
  //AUBIO_DBG("input_channels: %d\n", s->input_channels);

  if (samplerate == 0) {
    samplerate = s->input_samplerate;
    //AUBIO_DBG("sampling rate set to 0, automagically adjusting to %d\n", samplerate);
  }
  s->samplerate = samplerate;

  if (s->samplerate >  s->input_samplerate) {
    AUBIO_WRN("source_avcodec: upsampling %s from %d to %d\n", s->path,
        s->input_samplerate, s->samplerate);
  }

  AVFrame *avFrame = s->avFrame;
  avFrame = av_frame_alloc();
  if (!avFrame) {
    AUBIO_ERR("source_avcodec: Could not allocate frame for (%s)\n", s->path);
  }

  /* allocate output for avr */
  s->output = (float *)av_malloc(AUBIO_AVCODEC_MAX_BUFFER_SIZE * sizeof(float));

  s->read_samples = 0;
  s->read_index = 0;

  s->avFormatCtx = avFormatCtx;
  s->avCodecCtx = avCodecCtx;
  s->avFrame = avFrame;

  // default to mono output
  aubio_source_avcodec_reset_resampler(s, 0);

  s->eof = 0;
  s->multi = 0;

  //av_log_set_level(AV_LOG_QUIET);

  return s;

beach:
  //AUBIO_ERR("can not read %s at samplerate %dHz with a hop_size of %d\n",
  //    s->path, s->samplerate, s->hop_size);
  del_aubio_source_avcodec(s);
  return NULL;
}

void aubio_source_avcodec_reset_resampler(aubio_source_avcodec_t * s, uint_t multi) {
  if ( (multi != s->multi) || (s->avr == NULL) ) {
    int64_t input_layout = av_get_default_channel_layout(s->input_channels);
    uint_t output_channels = multi ? s->input_channels : 1;
    int64_t output_layout = av_get_default_channel_layout(output_channels);
    if (s->avr != NULL) {
      avresample_close( s->avr );
      av_free ( s->avr );
      s->avr = NULL;
    }
    AVAudioResampleContext *avr = s->avr;
    avr = avresample_alloc_context();

    av_opt_set_int(avr, "in_channel_layout",  input_layout,           0);
    av_opt_set_int(avr, "out_channel_layout", output_layout,          0);
    av_opt_set_int(avr, "in_sample_rate",     s->input_samplerate,    0);
    av_opt_set_int(avr, "out_sample_rate",    s->samplerate,          0);
    av_opt_set_int(avr, "in_sample_fmt",      s->avCodecCtx->sample_fmt, 0);
    av_opt_set_int(avr, "out_sample_fmt",     AV_SAMPLE_FMT_FLT,      0);
    int err;
    if ( ( err = avresample_open(avr) ) < 0) {
      char errorstr[256];
      av_strerror (err, errorstr, sizeof(errorstr));
      AUBIO_ERR("source_avcodec: Could not open AVAudioResampleContext for %s (%s)\n",
          s->path, errorstr);
      //goto beach;
      return;
    }
    s->avr = avr;
    s->multi = multi;
  }
}

void aubio_source_avcodec_readframe(aubio_source_avcodec_t *s, uint_t * read_samples) {
  AVFormatContext *avFormatCtx = s->avFormatCtx;
  AVCodecContext *avCodecCtx = s->avCodecCtx;
  AVFrame *avFrame = s->avFrame;
  AVPacket avPacket;
  av_init_packet (&avPacket);
  AVAudioResampleContext *avr = s->avr;
  float *output = s->output;
  *read_samples = 0;

  do
  {
    int err = av_read_frame (avFormatCtx, &avPacket);
    if (err == AVERROR_EOF) {
      s->eof = 1;
      goto beach;
    }
    if (err != 0) {
      char errorstr[256];
      av_strerror (err, errorstr, sizeof(errorstr));
      AUBIO_ERR("Could not read frame in %s (%s)\n", s->path, errorstr);
      goto beach;
    }
  } while (avPacket.stream_index != s->selected_stream);

  int got_frame = 0;
  int len = avcodec_decode_audio4(avCodecCtx, avFrame, &got_frame, &avPacket);

  if (len < 0) {
    AUBIO_ERR("Error while decoding %s\n", s->path);
    goto beach;
  }
  if (got_frame == 0) {
    //AUBIO_ERR("Could not get frame for (%s)\n", s->path);
    goto beach;
  }

  int in_linesize = 0;
  av_samples_get_buffer_size(&in_linesize, avCodecCtx->channels,
      avFrame->nb_samples, avCodecCtx->sample_fmt, 1);
  int in_samples = avFrame->nb_samples;
  int out_linesize = 0;
  int max_out_samples = AUBIO_AVCODEC_MAX_BUFFER_SIZE;
  int out_samples = avresample_convert ( avr,
        (uint8_t **)&output, out_linesize, max_out_samples,
        (uint8_t **)avFrame->data, in_linesize, in_samples);
  if (out_samples <= 0) {
    //AUBIO_ERR("No sample found while converting frame (%s)\n", s->path);
    goto beach;
  }

  *read_samples = out_samples;

beach:
  s->avFormatCtx = avFormatCtx;
  s->avCodecCtx = avCodecCtx;
  s->avFrame = avFrame;
  s->avr = avr;
  s->output = output;

  av_free_packet(&avPacket);
}

void aubio_source_avcodec_do(aubio_source_avcodec_t * s, fvec_t * read_data, uint_t * read){
  if (s->multi == 1) aubio_source_avcodec_reset_resampler(s, 0);
  uint_t i;
  uint_t end = 0;
  uint_t total_wrote = 0;
  while (total_wrote < s->hop_size) {
    end = MIN(s->read_samples - s->read_index, s->hop_size - total_wrote);
    for (i = 0; i < end; i++) {
      read_data->data[i + total_wrote] = s->output[i + s->read_index];
    }
    total_wrote += end;
    if (total_wrote < s->hop_size) {
      uint_t avcodec_read = 0;
      aubio_source_avcodec_readframe(s, &avcodec_read);
      s->read_samples = avcodec_read;
      s->read_index = 0;
      if (s->eof) {
        break;
      }
    } else {
      s->read_index += end;
    }
  }
  if (total_wrote < s->hop_size) {
    for (i = end; i < s->hop_size; i++) {
      read_data->data[i] = 0.;
    }
  }
  *read = total_wrote;
}

void aubio_source_avcodec_do_multi(aubio_source_avcodec_t * s, fmat_t * read_data, uint_t * read){
  if (s->multi == 0) aubio_source_avcodec_reset_resampler(s, 1);
  uint_t i,j;
  uint_t end = 0;
  uint_t total_wrote = 0;
  while (total_wrote < s->hop_size) {
    end = MIN(s->read_samples - s->read_index, s->hop_size - total_wrote);
    for (j = 0; j < read_data->height; j++) {
      for (i = 0; i < end; i++) {
        read_data->data[j][i + total_wrote] =
          s->output[(i + s->read_index) * s->input_channels + j];
      }
    }
    total_wrote += end;
    if (total_wrote < s->hop_size) {
      uint_t avcodec_read = 0;
      aubio_source_avcodec_readframe(s, &avcodec_read);
      s->read_samples = avcodec_read;
      s->read_index = 0;
      if (s->eof) {
        break;
      }
    } else {
      s->read_index += end;
    }
  }
  if (total_wrote < s->hop_size) {
    for (j = 0; j < read_data->height; j++) {
      for (i = end; i < s->hop_size; i++) {
        read_data->data[j][i] = 0.;
      }
    }
  }
  *read = total_wrote;
}

uint_t aubio_source_avcodec_get_samplerate(aubio_source_avcodec_t * s) {
  return s->samplerate;
}

uint_t aubio_source_avcodec_get_channels(aubio_source_avcodec_t * s) {
  return s->input_channels;
}

uint_t aubio_source_avcodec_seek (aubio_source_avcodec_t * s, uint_t pos) {
  int64_t resampled_pos = (uint_t)ROUND(pos * (s->input_samplerate * 1. / s->samplerate));
  int64_t min_ts = MAX(resampled_pos - 2000, 0);
  int64_t max_ts = MIN(resampled_pos + 2000, INT64_MAX);
  int seek_flags = AVSEEK_FLAG_FRAME | AVSEEK_FLAG_ANY;
  int ret = avformat_seek_file(s->avFormatCtx, s->selected_stream,
      min_ts, resampled_pos, max_ts, seek_flags);
  if (ret < 0) {
    AUBIO_ERR("Failed seeking to %d in file %s", pos, s->path);
  }
  // reset read status
  s->eof = 0;
  s->read_index = 0;
  s->read_samples = 0;
  // reset the AVAudioResampleContext
  avresample_close(s->avr);
  avresample_open(s->avr);
  return ret;
}

uint_t aubio_source_avcodec_close(aubio_source_avcodec_t * s) {
  if (s->avr != NULL) {
    avresample_close( s->avr );
    av_free ( s->avr );
  }
  s->avr = NULL;
  if (s->avCodecCtx != NULL) {
    avcodec_close ( s->avCodecCtx );
  }
  s->avCodecCtx = NULL;
  if (s->avFormatCtx != NULL) {
    avformat_close_input ( &(s->avFormatCtx) );
  }
  s->avFormatCtx = NULL;
  return AUBIO_OK;
}

void del_aubio_source_avcodec(aubio_source_avcodec_t * s){
  if (!s) return;
  aubio_source_avcodec_close(s);
  if (s->output != NULL) {
    av_free(s->output);
  }
  s->output = NULL;
  if (s->avFrame != NULL) {
    av_frame_free( &(s->avFrame) );
  }
  s->avFrame = NULL;
  AUBIO_FREE(s);
}

#endif /* HAVE_LIBAV */

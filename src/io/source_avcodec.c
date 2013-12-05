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

#ifdef HAVE_AVCODEC

#include <sndfile.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavresample/avresample.h>
#include <libavutil/opt.h>
#include <stdlib.h>

#include "aubio_priv.h"
#include "fvec.h"
#include "fmat.h"
#include "source_avcodec.h"

#define AUBIO_AVCODEC_MIN_BUFFER_SIZE FF_MIN_BUFFER_SIZE

#define SHORT_TO_FLOAT(x) (smpl_t)(x * 3.0517578125e-05)

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
  AVPacket avPacket;
  AVAudioResampleContext *avr;
  int16_t *output;
  uint_t read_samples;
  uint_t read_index;
};

aubio_source_avcodec_t * new_aubio_source_avcodec(char_t * path, uint_t samplerate, uint_t hop_size) {
  aubio_source_avcodec_t * s = AUBIO_NEW(aubio_source_avcodec_t);
  int err;

  if (path == NULL) {
    AUBIO_ERR("Aborted opening null path\n");
    return NULL;
  }

  s->hop_size = hop_size;
  s->channels = 1;
  s->path = path;

  // try opening the file and get some info about it
  // register all formats and codecs
  av_register_all();

  // open file
  AVFormatContext *avFormatCtx = s->avFormatCtx;
  avFormatCtx = NULL;
  if ( (err = avformat_open_input(&avFormatCtx, s->path, NULL, NULL) ) < 0 ) {
    uint8_t errorstr_len = 128;
    char errorstr[errorstr_len];
    if (av_strerror (err, errorstr, errorstr_len) == 0) {
      AUBIO_ERR("Failed opening %s (%s)\n", s->path, errorstr);
    } else {
      AUBIO_ERR("Failed opening %s (unknown error)\n", s->path);
    }
    goto beach;
  }

  // try to make sure max_analyze_duration is big enough for most songs
  avFormatCtx->max_analyze_duration *= 100;

  // retrieve stream information
  if ( (err = avformat_find_stream_info(avFormatCtx, NULL)) < 0 ) {
    uint8_t errorstr_len = 128;
    char errorstr[errorstr_len];
    if (av_strerror (err, errorstr, errorstr_len) == 0) {
      AUBIO_ERR("Could not find stream information for %s (%s)\n", s->path, errorstr);
    } else {
      AUBIO_ERR("Could not find stream information for %s (unknown error)\n", s->path);
    }
    goto beach;
  }

  // Dump information about file onto standard error
  //av_dump_format(avFormatCtx, 0, s->path, 0);

  uint_t i;
  sint_t selected_stream = -1;
  for (i = 0; i < avFormatCtx->nb_streams; i++) {
    if (avFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
      if (selected_stream == -1) {
        selected_stream = i;
      } else {
        AUBIO_WRN("More than one audio stream in %s, taking the first one\n", s->path);
      }
    }
  }
  if (selected_stream == -1) {
    AUBIO_ERR("No audio stream in %s\n", s->path);
    goto beach;
  }

  //AUBIO_DBG("Taking stream %d in file %s\n", selected_stream, s->path);

  AVCodecContext *avCodecCtx = s->avCodecCtx;
  avCodecCtx = avFormatCtx->streams[selected_stream]->codec;
  AVCodec *codec = avcodec_find_decoder(avCodecCtx->codec_id);
  if (codec == NULL) {
    AUBIO_ERR("Could not find decoder for %s", s->path);
    goto beach;
  }

  if ( ( err = avcodec_open2(avCodecCtx, codec, NULL) ) < 0) {
    uint8_t errorstr_len = 128;
    char errorstr[errorstr_len];
    if (av_strerror (err, errorstr, errorstr_len) == 0) {
      AUBIO_ERR("Could not load codec for %s (%s)\n", s->path, errorstr);
    } else {
      AUBIO_ERR("Could not load codec for %s (unknown error)\n", s->path);
    }
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

  int64_t input_layout = av_get_default_channel_layout(s->input_channels);
  int64_t mono_layout = av_get_default_channel_layout(1);

  AVAudioResampleContext *avr = s->avr;
  avr = avresample_alloc_context();
  av_opt_set_int(avr, "in_channel_layout",  input_layout,           0);
  av_opt_set_int(avr, "out_channel_layout", mono_layout,            0);
  av_opt_set_int(avr, "in_sample_rate",     s->input_samplerate,    0);
  av_opt_set_int(avr, "out_sample_rate",    s->samplerate,          0);
  av_opt_set_int(avr, "in_sample_fmt",      avCodecCtx->sample_fmt, 0);
  av_opt_set_int(avr, "out_sample_fmt",     AV_SAMPLE_FMT_S16,      0);
  if ( ( err = avresample_open(avr) ) < 0) {
    uint8_t errorstr_len = 128;
    char errorstr[errorstr_len];
    if (av_strerror (err, errorstr, errorstr_len) == 0) {
      AUBIO_ERR("Could not open AVAudioResampleContext for %s (%s)\n", s->path, errorstr);
    } else {
      AUBIO_ERR("Could not open AVAudioResampleContext for %s (unknown error)\n", s->path);
    }
    goto beach;
  }

  AVFrame *avFrame = s->avFrame;
  avFrame = avcodec_alloc_frame();
  if (!avFrame) {
    AUBIO_ERR("Could not allocate frame for (%s)\n", s->path);
  }
  AVPacket avPacket = s->avPacket;
  av_init_packet(&avPacket);

  /* allocate output for avr */
  s->output = (int16_t *)av_malloc(AUBIO_AVCODEC_MIN_BUFFER_SIZE * sizeof(int16_t));

  s->read_samples = 0;
  s->read_index = 0;

  s->avFormatCtx = avFormatCtx;
  s->avCodecCtx = avCodecCtx;
  s->avFrame = avFrame;
  s->avPacket = avPacket;
  s->avr = avr;

  //av_log_set_level(AV_LOG_QUIET);

  return s;

beach:
  AUBIO_ERR("can not read %s at samplerate %dHz with a hop_size of %d\n",
      s->path, s->samplerate, s->hop_size);
  del_aubio_source_avcodec(s);
  return NULL;
}

void aubio_source_avcodec_readframe(aubio_source_avcodec_t *s, uint_t * read_samples) {
  AVFormatContext *avFormatCtx = s->avFormatCtx;
  AVCodecContext *avCodecCtx = s->avCodecCtx;
  AVFrame *avFrame = s->avFrame;
  AVPacket avPacket = s->avPacket;
  AVAudioResampleContext *avr = s->avr;
  int16_t *output = s->output;

  uint_t i;
  int err = av_read_frame (avFormatCtx, &avPacket);
  if (err != 0) {
    //AUBIO_ERR("Could not read frame for (%s)\n", s->path);
    *read_samples = 0;
    return;
  }

  int got_frame = 0;
  int len = avcodec_decode_audio4(avCodecCtx, avFrame, &got_frame, &avPacket);

  if (len < 0) {
    AUBIO_ERR("Error while decoding %s\n", s->path);
    return;
  }
  if (got_frame == 0) {
    AUBIO_ERR("Could not get frame for (%s)\n", s->path);
  } /* else {
    int data_size =
      av_samples_get_buffer_size(NULL,
        avCodecCtx->channels, avFrame->nb_samples,
        avCodecCtx->sample_fmt, 1);
    AUBIO_WRN("Got data_size %d frame for (%s)\n", data_size, s->path);
  } */

  int in_samples = avFrame->nb_samples;
  int in_plane_size = 0; //avFrame->linesize[0];
  int out_plane_size = 0; //sizeof(float); //in_samples * sizeof(float);
  int max_out_samples = AUBIO_AVCODEC_MIN_BUFFER_SIZE;
  if (avresample_convert ( avr,
        (uint8_t **)&output, out_plane_size, max_out_samples,
        (uint8_t **)avFrame->data, in_plane_size, in_samples) < 0) {
      AUBIO_ERR("Could not convert frame  (%s)\n", s->path);
  }
  //AUBIO_ERR("Got in_plane_size %d frame for (%s)\n", in_plane_size, s->path);
  //AUBIO_WRN("Delay is %d for %s\n", avresample_get_delay(avr), s->path);
  //AUBIO_WRN("max_out_samples is %d for AUBIO_AVCODEC_MIN_BUFFER_SIZE %d\n",
  //    max_out_samples, AUBIO_AVCODEC_MIN_BUFFER_SIZE);

  uint_t out_samples = avresample_available(avr) + (avresample_get_delay(avr)
        + in_samples) * s->samplerate / s->input_samplerate;
  //AUBIO_WRN("Converted %d to %d samples\n", in_samples, out_samples);
  //for (i = 0; i < out_samples; i ++) {
  //  AUBIO_DBG("%f\n", SHORT_TO_FLOAT(output[i]));
  //}
  s->avFormatCtx = avFormatCtx;
  s->avCodecCtx = avCodecCtx;
  s->avFrame = avFrame;
  s->avPacket = avPacket;
  s->avr = avr;
  s->output = output;

  *read_samples = out_samples;
}


void aubio_source_avcodec_do(aubio_source_avcodec_t * s, fvec_t * read_data, uint_t * read){
  uint_t i;
  //AUBIO_DBG("entering 'do' with %d, %d\n", s->read_samples, s->read_index);
  // begin reading
  if (s->read_samples == 0) {
    uint_t avcodec_read = 0;
    aubio_source_avcodec_readframe(s, &avcodec_read);
    s->read_samples += avcodec_read;
    s->read_index = 0;
  }
  if (s->read_samples < s->hop_size) {
    // write the end of the buffer to the beginning of read_data
    uint_t partial = s->read_samples;
    for (i = 0; i < partial; i++) {
      read_data->data[i] = SHORT_TO_FLOAT(s->output[i + s->read_index]);
    }
    s->read_samples = 0;
    s->read_index = 0;
    // get more data
    uint_t avcodec_read = 0;
    aubio_source_avcodec_readframe(s, &avcodec_read);
    s->read_samples += avcodec_read;
    s->read_index = 0;
    // write the beginning of the buffer to the end of read_data
    uint_t end = MIN(s->hop_size, s->read_samples);
    if (avcodec_read == 0) {
      end = partial;
    }
    for (i = partial; i < end; i++) {
      read_data->data[i] = SHORT_TO_FLOAT(s->output[i - partial + s->read_index]);
    }
    if (end < s->hop_size) {
      for (i = end; i < s->hop_size; i++) {
        read_data->data[i] = 0.;
      }
    }
    s->read_index += partial;
    s->read_samples -= partial;
    *read = end;
  } else {
    for (i = 0; i < s->hop_size; i++) {
      read_data->data[i] = SHORT_TO_FLOAT(s->output[i + s->read_index]);
    }
    s->read_index += s->hop_size;
    s->read_samples -= s->hop_size;
    *read = s->hop_size;
  }
}

void aubio_source_avcodec_do_multi(aubio_source_avcodec_t * s, fmat_t * read_data, uint_t * read){
  //uint_t i,j, input_channels = s->input_channels;
}

uint_t aubio_source_avcodec_get_samplerate(aubio_source_avcodec_t * s) {
  return s->samplerate;
}

uint_t aubio_source_avcodec_get_channels(aubio_source_avcodec_t * s) {
  return s->input_channels;
}

uint_t aubio_source_avcodec_seek (aubio_source_avcodec_t * s, uint_t pos) {
  //uint_t resampled_pos = (uint_t)ROUND(pos * s->input_samplerate * 1. / s->samplerate);
  return 0; //sf_seek (s->handle, resampled_pos, SEEK_SET);
}

void del_aubio_source_avcodec(aubio_source_avcodec_t * s){
  if (!s) return;
  if (s->output != NULL) {
    av_free(s->output);
  }
  if (s->avr != NULL) {
    avresample_close( s->avr );
    av_free ( s->avr );
  }
  s->avr = NULL;
  if (s->avFrame != NULL) {
    avcodec_free_frame( &(s->avFrame) );
  }
  s->avFrame = NULL;
  if ( &(s->avPacket) != NULL) {
    av_free_packet( &(s->avPacket) );
  }
  if (s->avCodecCtx != NULL) {
    avcodec_close ( s->avCodecCtx );
  }
  s->avCodecCtx = NULL;
  if (s->avFormatCtx != NULL) {
    avformat_close_input ( &(s->avFormatCtx) );
  }
  s->avFrame = NULL;
  s->avFormatCtx = NULL;
  AUBIO_FREE(s);
}

#endif /* HAVE_SNDFILE */

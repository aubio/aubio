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

#ifdef __APPLE__

#include "aubio_priv.h"
#include "fvec.h"
#include "io/sink_apple_audio.h"

// CFURLRef, CFURLCreateWithFileSystemPath, ...
#include <CoreFoundation/CoreFoundation.h>
// ExtAudioFileRef, AudioStreamBasicDescription, AudioBufferList, ...
#include <AudioToolbox/AudioToolbox.h>

#define FLOAT_TO_SHORT(x) (short)(x * 32768)

extern int createAubioBufferList(AudioBufferList *bufferList, int channels, int segmentSize);
extern void freeAudioBufferList(AudioBufferList *bufferList);
extern CFURLRef getURLFromPath(const char * path);
char_t *getPrintableOSStatusError(char_t *str, OSStatus error);

#define MAX_SIZE 4096 // the maximum number of frames that can be written at a time

struct _aubio_sink_apple_audio_t {
  uint_t samplerate;
  uint_t channels;
  char_t *path;

  uint_t max_frames;

  AudioBufferList bufferList;
  ExtAudioFileRef audioFile;
  bool async;
};

aubio_sink_apple_audio_t * new_aubio_sink_apple_audio(char_t * uri, uint_t samplerate) {
  aubio_sink_apple_audio_t * s = AUBIO_NEW(aubio_sink_apple_audio_t);
  s->samplerate = samplerate;
  s->channels = 1;
  s->path = uri;
  s->max_frames = MAX_SIZE;
  s->async = true;

  AudioStreamBasicDescription clientFormat;
  memset(&clientFormat, 0, sizeof(AudioStreamBasicDescription));
  clientFormat.mFormatID         = kAudioFormatLinearPCM;
  clientFormat.mSampleRate       = (Float64)(s->samplerate);
  clientFormat.mFormatFlags      = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
  clientFormat.mChannelsPerFrame = s->channels;
  clientFormat.mBitsPerChannel   = sizeof(short) * 8;
  clientFormat.mFramesPerPacket  = 1;
  clientFormat.mBytesPerFrame    = clientFormat.mBitsPerChannel * clientFormat.mChannelsPerFrame / 8;
  clientFormat.mBytesPerPacket   = clientFormat.mFramesPerPacket * clientFormat.mBytesPerFrame;
  clientFormat.mReserved         = 0;

  AudioFileTypeID fileType = kAudioFileWAVEType;
  CFURLRef fileURL = getURLFromPath(uri);
  bool overwrite = true;
  OSStatus err = noErr;
  err = ExtAudioFileCreateWithURL(fileURL, fileType, &clientFormat, NULL,
     overwrite ? kAudioFileFlags_EraseFile : 0, &s->audioFile);
  if (err) {
    char_t errorstr[20];
    AUBIO_ERR("sink_apple_audio: error when trying to create %s with "
        "ExtAudioFileCreateWithURL (%s)\n", s->path,
        getPrintableOSStatusError(errorstr, err));
    goto beach;
  }
  if (createAubioBufferList(&s->bufferList, s->channels, s->max_frames * s->channels)) {
    AUBIO_ERR("sink_apple_audio: error when creating buffer list for %s, "
        "out of memory? \n", s->path);
    goto beach;
  }
  return s;

beach:
  AUBIO_FREE(s);
  return NULL;
}

void aubio_sink_apple_audio_do(aubio_sink_apple_audio_t * s, fvec_t * write_data, uint_t write) {
  OSStatus err = noErr;
  UInt32 c, v;
  short *data = (short*)s->bufferList.mBuffers[0].mData;
  if (write > s->max_frames) {
    AUBIO_WRN("sink_apple_audio: trying to write %d frames, max %d\n", write, s->max_frames);
    write = s->max_frames;
  }
  smpl_t *buf = write_data->data;

  if (buf) {
      for (c = 0; c < s->channels; c++) {
          for (v = 0; v < write; v++) {
              data[v * s->channels + c] =
                  FLOAT_TO_SHORT(buf[ v * s->channels + c]);
          }
      }
  }
  if (s->async) {
    err = ExtAudioFileWriteAsync(s->audioFile, write, &s->bufferList);

    if (err) {
      char_t errorstr[20];
      AUBIO_ERROR("sink_apple_audio: error while writing %s "
          "in ExtAudioFileWriteAsync (%s), switching to sync\n", s->path,
          getPrintableOSStatusError(errorstr, err));
      s->async = false;
    } else {
      return;
    }

  } else {
    err = ExtAudioFileWrite(s->audioFile, write, &s->bufferList);

    if (err) {
      char_t errorstr[20];
      AUBIO_ERROR("sink_apple_audio: error while writing %s "
          "in ExtAudioFileWrite (%s)\n", s->path,
          getPrintableOSStatusError(errorstr, err));
    }
  }
  return;
}

uint_t aubio_sink_apple_audio_close(aubio_sink_apple_audio_t * s) {
  OSStatus err = noErr;
  if (!s->audioFile) {
    return AUBIO_FAIL;
  }
  err = ExtAudioFileDispose(s->audioFile);
  if (err) {
    char_t errorstr[20];
    AUBIO_ERROR("sink_apple_audio: error while closing %s "
        "in ExtAudioFileDispose (%s)\n", s->path,
        getPrintableOSStatusError(errorstr, err));
  }
  s->audioFile = NULL;
  return err;
}

void del_aubio_sink_apple_audio(aubio_sink_apple_audio_t * s) {
  if (s->audioFile) aubio_sink_apple_audio_close (s);
  freeAudioBufferList(&s->bufferList);
  AUBIO_FREE(s);
  return;
}

#endif /* __APPLE__ */

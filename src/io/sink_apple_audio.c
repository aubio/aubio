/*
  Copyright (C) 2012-2014 Paul Brossier <piem@aubio.org>

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

#ifdef HAVE_SINK_APPLE_AUDIO
#include "fvec.h"
#include "fmat.h"
#include "io/sink_apple_audio.h"
#include "io/ioutils.h"

// CFURLRef, CFURLCreateWithFileSystemPath, ...
#include <CoreFoundation/CoreFoundation.h>
// ExtAudioFileRef, AudioStreamBasicDescription, AudioBufferList, ...
#include <AudioToolbox/AudioToolbox.h>

extern int createAudioBufferList(AudioBufferList *bufferList, int channels, int segmentSize);
extern void freeAudioBufferList(AudioBufferList *bufferList);
extern CFURLRef createURLFromPath(const char * path);
char_t *getPrintableOSStatusError(char_t *str, OSStatus error);

uint_t aubio_sink_apple_audio_open(aubio_sink_apple_audio_t *s);

uint_t aubio_str_extension_matches(const char_t *ext,
    const char_t *pattern);
const char_t *aubio_str_get_extension(const char_t *filename);

#define MAX_SIZE 4096 // the maximum number of frames that can be written at a time

void aubio_sink_apple_audio_write(aubio_sink_apple_audio_t *s, uint_t write);

struct _aubio_sink_apple_audio_t {
  uint_t samplerate;
  uint_t channels;
  char_t *path;

  uint_t max_frames;

  AudioBufferList bufferList;
  ExtAudioFileRef audioFile;
  bool async;
  AudioFileTypeID fileType;
};

aubio_sink_apple_audio_t * new_aubio_sink_apple_audio(const char_t * uri, uint_t samplerate) {
  aubio_sink_apple_audio_t * s = AUBIO_NEW(aubio_sink_apple_audio_t);
  s->max_frames = MAX_SIZE;
  s->async = false;

  if ( (uri == NULL) || (strnlen(uri, PATH_MAX) < 1) ) {
    AUBIO_ERROR("sink_apple_audio: Aborted opening null path\n");
    goto beach;
  }

  s->path = AUBIO_ARRAY(char_t, strnlen(uri, PATH_MAX) + 1);
  strncpy(s->path, uri, strnlen(uri, PATH_MAX) + 1);

  s->samplerate = 0;
  s->channels = 0;

  aubio_sink_apple_audio_preset_format(s, aubio_str_get_extension(uri));

  // zero samplerate given. do not open yet
  if ((sint_t)samplerate == 0) {
    return s;
  }

  // invalid samplerate given, abort
  if (aubio_io_validate_samplerate("sink_apple_audio", s->path, samplerate)) {
    goto beach;
  }

  s->samplerate = samplerate;
  s->channels = 1;

  if (aubio_sink_apple_audio_open(s) != AUBIO_OK) {
    // open failed, abort
    goto beach;
  }

  return s;
beach:
  del_aubio_sink_apple_audio(s);
  return NULL;
}

uint_t aubio_sink_apple_audio_preset_samplerate(aubio_sink_apple_audio_t *s, uint_t samplerate)
{
  if (aubio_io_validate_samplerate("sink_apple_audio", s->path, samplerate)) {
    return AUBIO_FAIL;
  }
  s->samplerate = samplerate;
  // automatically open when both samplerate and channels have been set
  if (/* s->samplerate != 0 && */ s->channels != 0) {
    return aubio_sink_apple_audio_open(s);
  }
  return AUBIO_OK;
}

uint_t aubio_sink_apple_audio_preset_channels(aubio_sink_apple_audio_t *s, uint_t channels)
{
  if (aubio_io_validate_channels("sink_apple_audio", s->path, channels)) {
    return AUBIO_FAIL;
  }
  s->channels = channels;
  // automatically open when both samplerate and channels have been set
  if (s->samplerate != 0 /* && s->channels != 0 */) {
    return aubio_sink_apple_audio_open(s);
  }
  return AUBIO_OK;
}

uint_t aubio_sink_apple_audio_preset_format(aubio_sink_apple_audio_t *s,
    const char_t *fmt)
{
  if (aubio_str_extension_matches(fmt, "wav")) {
    s->fileType = kAudioFileWAVEType;
  } else if (aubio_str_extension_matches(fmt, "m4a")
      || aubio_str_extension_matches(fmt, "mp4") ) {
    // use alac for "mp4" and "m4a"
    s->fileType = kAudioFileM4AType;
  } else if (aubio_str_extension_matches(fmt, "aac") ) {
    // only use lossy codec for "aac"
    s->fileType = kAudioFileMPEG4Type;
  } else if (aubio_str_extension_matches(fmt, "aiff") ) {
    // only use lossy codec for "aac"
    s->fileType = kAudioFileAIFFType;
  } else {
    s->fileType = kAudioFileWAVEType;
    if (fmt && strnlen(fmt, PATH_MAX)) {
      AUBIO_WRN("sink_apple_audio: could not guess format for %s,"
         " using default (wav)\n", s->path);
      return AUBIO_FAIL;
    }
  }
  return AUBIO_OK;
}

static void aubio_sink_apple_audio_set_client_format(aubio_sink_apple_audio_t* s,
    AudioStreamBasicDescription *clientFormat)
{
  memset(clientFormat, 0, sizeof(AudioStreamBasicDescription));
  // always set samplerate and channels first
  clientFormat->mSampleRate       = (Float64)(s->samplerate);
  clientFormat->mChannelsPerFrame = s->channels;

  switch (s->fileType) {
    case kAudioFileM4AType:
      clientFormat->mFormatID         = kAudioFormatAppleLossless;
      break;
    case kAudioFileMPEG4Type:
      clientFormat->mFormatID         = kAudioFormatMPEG4AAC;
      clientFormat->mFormatFlags      = kMPEG4Object_AAC_Main;
      clientFormat->mFormatFlags     |= kAppleLosslessFormatFlag_16BitSourceData;
      clientFormat->mFramesPerPacket  = 1024;
      break;
    case kAudioFileWAVEType:
      clientFormat->mFormatID         = kAudioFormatLinearPCM;
      clientFormat->mFormatFlags      = kAudioFormatFlagIsSignedInteger;
      clientFormat->mFormatFlags     |= kAudioFormatFlagIsPacked;
      clientFormat->mBitsPerChannel   = sizeof(short) * 8;
      clientFormat->mFramesPerPacket  = 1;
      clientFormat->mBytesPerFrame    = clientFormat->mBitsPerChannel * clientFormat->mChannelsPerFrame / 8;
      clientFormat->mBytesPerPacket   = clientFormat->mFramesPerPacket * clientFormat->mBytesPerFrame;
      break;
    case kAudioFileAIFFType:
      clientFormat->mFormatID         = kAudioFormatLinearPCM;
      clientFormat->mFormatFlags      = kAudioFormatFlagIsSignedInteger;
      clientFormat->mFormatFlags     |= kAudioFormatFlagIsPacked;
      clientFormat->mFormatFlags     |= kAudioFormatFlagIsBigEndian;
      clientFormat->mBitsPerChannel   = sizeof(short) * 8;
      clientFormat->mFramesPerPacket  = 1;
      clientFormat->mBytesPerFrame    = clientFormat->mBitsPerChannel * clientFormat->mChannelsPerFrame / 8;
      clientFormat->mBytesPerPacket   = clientFormat->mFramesPerPacket * clientFormat->mBytesPerFrame;
      break;
    default:
      break;
  }
}

uint_t aubio_sink_apple_audio_get_samplerate(const aubio_sink_apple_audio_t *s)
{
  return s->samplerate;
}

uint_t aubio_sink_apple_audio_get_channels(const aubio_sink_apple_audio_t *s)
{
  return s->channels;
}

uint_t aubio_sink_apple_audio_open(aubio_sink_apple_audio_t *s) {

  if (s->samplerate == 0 || s->channels == 0) return AUBIO_FAIL;

  CFURLRef fileURL = createURLFromPath(s->path);
  bool overwrite = true;

  // set the in-memory format
  AudioStreamBasicDescription inputFormat;
  memset(&inputFormat, 0, sizeof(AudioStreamBasicDescription));
  inputFormat.mFormatID         = kAudioFormatLinearPCM;
  inputFormat.mSampleRate       = (Float64)(s->samplerate);
  inputFormat.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  inputFormat.mChannelsPerFrame = s->channels;
  inputFormat.mBitsPerChannel   = sizeof(smpl_t) * 8;
  inputFormat.mFramesPerPacket  = 1;
  inputFormat.mBytesPerFrame    = inputFormat.mBitsPerChannel * inputFormat.mChannelsPerFrame / 8;
  inputFormat.mBytesPerPacket   = inputFormat.mFramesPerPacket * inputFormat.mBytesPerFrame;

  // get the in-file format
  AudioStreamBasicDescription clientFormat;
  aubio_sink_apple_audio_set_client_format(s, &clientFormat);

  OSStatus err = noErr;
  err = ExtAudioFileCreateWithURL(fileURL, s->fileType, &clientFormat, NULL,
     overwrite ? kAudioFileFlags_EraseFile : 0, &s->audioFile);
  CFRelease(fileURL);
  if (err) {
    char_t errorstr[20];
    AUBIO_ERR("sink_apple_audio: error when trying to create %s with "
        "ExtAudioFileCreateWithURL (%s)\n", s->path,
        getPrintableOSStatusError(errorstr, err));
    goto beach;
  }

#if defined(kAppleSoftwareAudioCodecManufacturer)
  // on iOS, set software based encoding before setting clientDataFormat
  UInt32 codecManf = kAppleSoftwareAudioCodecManufacturer;
  err = ExtAudioFileSetProperty(s->audioFile,
      kExtAudioFileProperty_CodecManufacturer,
      sizeof(UInt32), &codecManf);
  if (err) {
    char_t errorstr[20];
    AUBIO_ERR("sink_apple_audio: error when trying to set sofware codec on %s "
        "(%s)\n", s->path, getPrintableOSStatusError(errorstr, err));
    goto beach;
  }
#endif

  err = ExtAudioFileSetProperty(s->audioFile,
      kExtAudioFileProperty_ClientDataFormat,
      sizeof(AudioStreamBasicDescription), &inputFormat);
  if (err) {
    char_t errorstr[20];
    AUBIO_ERR("sink_apple_audio: error when trying to set output format on %s "
        "(%s)\n", s->path, getPrintableOSStatusError(errorstr, err));
    goto beach;
  }

  if (createAudioBufferList(&s->bufferList, s->channels, s->max_frames * s->channels)) {
    AUBIO_ERR("sink_apple_audio: error when creating buffer list for %s, "
        "out of memory? \n", s->path);
    goto beach;
  }
  return AUBIO_OK;

beach:
  return AUBIO_FAIL;
}

void aubio_sink_apple_audio_do(aubio_sink_apple_audio_t * s, fvec_t * write_data, uint_t write) {
  UInt32 c, v;
  smpl_t *data = (smpl_t*)s->bufferList.mBuffers[0].mData;
  uint_t length = aubio_sink_validate_input_length("sink_apple_audio", s->path,
      s->max_frames, write_data->length, write);

  for (c = 0; c < s->channels; c++) {
    for (v = 0; v < length; v++) {
      data[v * s->channels + c] = write_data->data[v];
    }
  }

  aubio_sink_apple_audio_write(s, length);
}

void aubio_sink_apple_audio_do_multi(aubio_sink_apple_audio_t * s, fmat_t * write_data, uint_t write) {
  UInt32 c, v;
  smpl_t *data = (smpl_t*)s->bufferList.mBuffers[0].mData;
  uint_t channels = aubio_sink_validate_input_channels("sink_apple_audio",
      s->path, s->channels, write_data->height);
  uint_t length = aubio_sink_validate_input_length("sink_apple_audio", s->path,
      s->max_frames, write_data->length, write);

  for (c = 0; c < channels; c++) {
    for (v = 0; v < length; v++) {
      data[v * s->channels + c] = write_data->data[c][v];
    }
  }

  aubio_sink_apple_audio_write(s, length);
}

void aubio_sink_apple_audio_write(aubio_sink_apple_audio_t *s, uint_t write) {
  OSStatus err = noErr;
  // set mDataByteSize to match the number of frames to be written
  // see https://www.mail-archive.com/coreaudio-api@lists.apple.com/msg01109.html
  s->bufferList.mBuffers[0].mDataByteSize = write * s->channels
    * sizeof(smpl_t);
  if (s->async) {
    err = ExtAudioFileWriteAsync(s->audioFile, write, &s->bufferList);
    if (err) {
      char_t errorstr[20];
      if (err == kExtAudioFileError_AsyncWriteBufferOverflow) {
        snprintf(errorstr, sizeof (errorstr), "buffer overflow");
      } else if (err == kExtAudioFileError_AsyncWriteTooLarge) {
        snprintf(errorstr, sizeof (errorstr), "write too large");
      } else {
        // unknown error
        getPrintableOSStatusError(errorstr, err);
      }
      AUBIO_ERROR("sink_apple_audio: error while writing %s "
                  "in ExtAudioFileWriteAsync (%s)\n", s->path, errorstr);
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
  AUBIO_ASSERT(s);
  if (s->audioFile)
    aubio_sink_apple_audio_close (s);
  if (s->path)
    AUBIO_FREE(s->path);
  freeAudioBufferList(&s->bufferList);
  AUBIO_FREE(s);
}

#endif /* HAVE_SINK_APPLE_AUDIO */

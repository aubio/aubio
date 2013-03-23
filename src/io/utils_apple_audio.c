#ifdef __APPLE__

// CFURLRef, CFURLCreateWithFileSystemPath, ...
#include <CoreFoundation/CoreFoundation.h>
// ExtAudioFileRef, AudioStreamBasicDescription, AudioBufferList, ...
#include <AudioToolbox/AudioToolbox.h>
#include "aubio_priv.h"

int createAubioBufferList(AudioBufferList *bufferList, int channels, int segmentSize);
void freeAudioBufferList(AudioBufferList *bufferList);
CFURLRef getURLFromPath(const char * path);

int createAubioBufferList(AudioBufferList * bufferList, int channels, int max_source_samples) {
  bufferList->mNumberBuffers = 1;
  bufferList->mBuffers[0].mNumberChannels = channels;
  bufferList->mBuffers[0].mData = AUBIO_ARRAY(short, max_source_samples);
  bufferList->mBuffers[0].mDataByteSize = max_source_samples * sizeof(short);
  return 0;
}

void freeAudioBufferList(AudioBufferList *bufferList) {
  UInt32 i = 0;
  if (!bufferList) return;
  for (i = 0; i < bufferList->mNumberBuffers; i++) {
    if (bufferList->mBuffers[i].mData) {
      AUBIO_FREE(bufferList->mBuffers[i].mData);
      bufferList->mBuffers[i].mData = NULL;
    }
  }
  bufferList = NULL;
}

CFURLRef getURLFromPath(const char * path) {
  CFStringRef cfTotalPath = CFStringCreateWithCString (kCFAllocatorDefault,
      path, kCFStringEncodingUTF8);

  return CFURLCreateWithFileSystemPath(kCFAllocatorDefault, cfTotalPath,
      kCFURLPOSIXPathStyle, false);
}

#endif /* __APPLE__ */

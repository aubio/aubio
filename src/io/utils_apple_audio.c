#ifdef __APPLE__

// CFURLRef, CFURLCreateWithFileSystemPath, ...
#include <CoreFoundation/CoreFoundation.h>
// ExtAudioFileRef, AudioStreamBasicDescription, AudioBufferList, ...
#include <AudioToolbox/AudioToolbox.h>

int createAubioBufferList(AudioBufferList *bufferList, int channels, int segmentSize);
void freeAudioBufferList(AudioBufferList *bufferList);
CFURLRef getURLFromPath(const char * path);

int createAubioBufferList(AudioBufferList * bufferList, int channels, int segmentSize) {
  bufferList->mNumberBuffers = 1;
  bufferList->mBuffers[0].mNumberChannels = channels;
  bufferList->mBuffers[0].mData = (short *)malloc(segmentSize * sizeof(short));
  bufferList->mBuffers[0].mDataByteSize = segmentSize * sizeof(short);
  return 0;
}

void freeAudioBufferList(AudioBufferList *bufferList) {
  UInt32 i = 0;
  if (!bufferList) return;
  for (i = 0; i < bufferList->mNumberBuffers; i++) {
    if (bufferList->mBuffers[i].mData) {
      free (bufferList->mBuffers[i].mData);
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

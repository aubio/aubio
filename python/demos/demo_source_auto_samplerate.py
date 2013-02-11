#! /usr/bin/env python

import sys
from aubio import source

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print 'usage: %s <inputfile>' % sys.argv[0]
    sys.exit(1)
  f = source(sys.argv[1], 0, 256)
  samplerate = f.get_samplerate()
  total_frames, read = 0, 256
  while read:
    vec, read = f()
    total_frames += read
  print f.uri, "is",
  print "%.2f seconds long at %.1fkHz" % (total_frames / float(samplerate), samplerate / 1000. )

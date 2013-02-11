#! /usr/bin/env python

import sys
from aubio import source

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print 'usage: %s <inputfile>' % sys.argv[0]
    sys.exit(1)
  f = source(sys.argv[1], 1, 256)
  samplerate = f.get_samplerate()
  total_frames, read = 0, 256
  while read:
    vec, read = f()
    total_frames += read
  print "read", total_frames / float(samplerate), "seconds from", f.uri

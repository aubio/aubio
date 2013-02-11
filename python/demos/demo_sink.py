#! /usr/bin/env python

import sys
from aubio import source, sink

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print 'usage: %s <inputfile> <outputfile>' % sys.argv[0]
    sys.exit(1)
  f = source(sys.argv[1], 8000, 256)
  g = sink(sys.argv[2], 8000)
  total_frames, read = 0, 256
  while read:
    vec, read = f()
    g(vec, read)
    total_frames += read
  print "read", total_frames / float(f.samplerate), "seconds from", f.uri

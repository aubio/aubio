#! /usr/bin/env python

import sys
from aubio import source

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: %s <inputfile> [samplerate] [hop_size]' % sys.argv[0]
        sys.exit(1)
    samplerate = 0
    hop_size = 256
    if len(sys.argv) > 2: samplerate = int(sys.argv[2])
    if len(sys.argv) > 3: hop_size = int(sys.argv[3])

    f = source(sys.argv[1], samplerate, hop_size)
    samplerate = f.samplerate

    total_frames, read = 0, f.hop_size
    while read:
        vec, read = f()
        total_frames += read
        if read < f.hop_size: break
    print "read", "%.2fs" % (total_frames / float(samplerate) ),
    print "(", total_frames, "frames", "in",
    print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
    print "from", f.uri

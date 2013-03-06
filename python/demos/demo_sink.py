#! /usr/bin/env python

import sys
from aubio import source, sink

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'usage: %s <inputfile> <outputfile> [samplerate] [hop_size]' % sys.argv[0]
        sys.exit(1)

    if len(sys.argv) > 3: samplerate = int(sys.argv[3])
    else: samplerate = 0
    if len(sys.argv) > 4: hop_size = int(sys.argv[4])
    else: hop_size = 256

    f = source(sys.argv[1], samplerate, hop_size)
    if samplerate == 0: samplerate = f.samplerate
    g = sink(sys.argv[2], samplerate)

    total_frames, read = 0, hop_size
    while read:
        vec, read = f()
        g(vec, read)
        total_frames += read
    print "wrote", "%.2fs" % (total_frames / float(samplerate) ),
    print "(", total_frames, "frames", "in",
    print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
    print "from", f.uri,
    print "to", g.uri

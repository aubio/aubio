#! /usr/bin/env python

import sys
import os.path
from aubio import source, sink

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'usage: %s <inputfile> <duration>' % sys.argv[0]
        sys.exit(1)
    source_file = sys.argv[1]
    duration = float(sys.argv[2])
    sink_base, sink_ext = os.path.splitext(os.path.basename(source_file))
    slice_n, total_frames, read = 1, 0, 256
    f = source(source_file, 0, 256)
    g = sink(sink_base + '-%02d' % slice_n + sink_ext, f.samplerate)
    while read:
        vec, read = f()
        g(vec, read)
        total_frames += read
        if total_frames / float(f.samplerate) >= duration * slice_n: 
            slice_n += 1
            del g
            g = sink(sink_base + '-%02d' % slice_n + sink_ext, f.samplerate)
    total_duration = total_frames / float(f.samplerate)
    print 'created %(slice_n)d slices from %(source_file)s' % locals(),
    print ' (total duration %(total_duration).2fs)' % locals()
    del f, g

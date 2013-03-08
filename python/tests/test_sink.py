#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, source, sink
from numpy import array
from utils import list_all_sounds

list_of_sounds = list_all_sounds('sounds')
path = None

class aubio_sink_test_case(TestCase):

    def setUp(self):
        if not len(list_of_sounds): self.skipTest('add some sound files in \'python/tests/sounds\'')

    def test_many_sinks(self):
        for i in range(100):
            g = sink('/tmp/f.wav', 0)
            write = 256
            for n in range(200):
                vec = fvec(write)
                g(vec, write)
            del g

    def test_read(self):
        for path in list_of_sounds:
            for samplerate, hop_size in zip([0, 44100, 8000, 32000], [512, 1024, 64, 256]):
                f = source(path, samplerate, hop_size)
                if samplerate == 0: samplerate = f.samplerate
                g = sink('/tmp/f.wav', samplerate)
                total_frames = 0
                while True:
                    vec, read = f()
                    g(vec, read)
                    total_frames += read
                    if read < f.hop_size: break
                print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
                print "(", total_frames, "frames", "in",
                print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
                print "from", f.uri,
                print "to", g.uri
                #del f, g

if __name__ == '__main__':
    from unittest import main
    main()

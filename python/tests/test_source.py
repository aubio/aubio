#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, source
from numpy import array
from utils import list_all_sounds

list_of_sounds = list_all_sounds('sounds')
path = None

class aubio_source_test_case(TestCase):

    def setUp(self):
        if not len(list_of_sounds): self.skipTest('add some sound files in \'python/tests/sounds\'')

    def read_from_sink(self, f):
        total_frames = 0
        while True:
            vec, read = f()
            total_frames += read
            if read < f.hop_size: break
        print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
        print "(", total_frames, "frames", "in",
        print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
        print "from", f.uri

    def test_samplerate_hopsize(self):
        for p in list_of_sounds:
            for samplerate, hop_size in zip([0, 44100, 8000, 32000], [ 512, 512, 64, 256]):
                f = source(p, samplerate, hop_size)
                assert f.samplerate != 0
                self.read_from_sink(f)

    def test_samplerate_none(self):
        for p in list_of_sounds:
            f = source(p)
            assert f.samplerate != 0
            self.read_from_sink(f)

    def test_samplerate_0(self):
        for p in list_of_sounds:
            f = source(p, 0)
            assert f.samplerate != 0
            self.read_from_sink(f)

    def test_wrong_samplerate(self):
        for p in list_of_sounds:
            try:
                f = source(p, -1)
            except Exception, e:
                print e
            else:
                self.fail('does not fail with wrong samplerate')

    def test_wrong_hop_size(self):
        for p in list_of_sounds:
            try:
                f = source(p, 0, -1)
            except Exception, e:
                print e
            else:
                self.fail('does not fail with wrong hop_size %d' % f.hop_size)

    def test_zero_hop_size(self):
        for p in list_of_sounds:
            f = source(p, 0, 0)
            assert f.samplerate != 0
            assert f.hop_size != 0
            self.read_from_sink(f)

if __name__ == '__main__':
    from unittest import main
    main()

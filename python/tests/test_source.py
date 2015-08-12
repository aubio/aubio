#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, source
from numpy import array
from utils import list_all_sounds

list_of_sounds = list_all_sounds('sounds')
path = None

class aubio_source_test_case_base(TestCase):

    def setUp(self):
        if not len(list_of_sounds): self.skipTest('add some sound files in \'python/tests/sounds\'')

class aubio_source_test_case(aubio_source_test_case_base):

    def test_close_file(self):
        samplerate = 0 # use native samplerate
        hop_size = 256
        for p in list_of_sounds:
            f = source(p, samplerate, hop_size)
            f.close()

    def test_close_file_twice(self):
        samplerate = 0 # use native samplerate
        hop_size = 256
        for p in list_of_sounds:
            f = source(p, samplerate, hop_size)
            f.close()
            f.close()

class aubio_source_read_test_case(aubio_source_test_case_base):

    def read_from_source(self, f):
        total_frames = 0
        while True:
            vec, read = f()
            total_frames += read
            if read < f.hop_size: break
        print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
        print "(", total_frames, "frames", "in",
        print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
        print "from", f.uri
        return total_frames

    def test_samplerate_hopsize(self):
        for p in list_of_sounds:
            for samplerate, hop_size in zip([0, 44100, 8000, 32000], [ 512, 512, 64, 256]):
                f = source(p, samplerate, hop_size)
                assert f.samplerate != 0
                self.read_from_source(f)

    def test_samplerate_none(self):
        for p in list_of_sounds:
            f = source(p)
            assert f.samplerate != 0
            self.read_from_source(f)

    def test_samplerate_0(self):
        for p in list_of_sounds:
            f = source(p, 0)
            assert f.samplerate != 0
            self.read_from_source(f)

    def test_wrong_samplerate(self):
        for p in list_of_sounds:
            try:
                f = source(p, -1)
            except ValueError, e:
                pass
            else:
                self.fail('negative samplerate does not raise ValueError')

    def test_wrong_hop_size(self):
        for p in list_of_sounds:
            try:
                f = source(p, 0, -1)
            except ValueError, e:
                pass
            else:
                self.fail('negative hop_size does not raise ValueError')

    def test_zero_hop_size(self):
        for p in list_of_sounds:
            f = source(p, 0, 0)
            assert f.samplerate != 0
            assert f.hop_size != 0
            self.read_from_source(f)

    def test_seek_to_half(self):
        from random import randint
        for p in list_of_sounds:
            f = source(p, 0, 0)
            assert f.samplerate != 0
            assert f.hop_size != 0
            a = self.read_from_source(f)
            c = randint(0, a)
            f.seek(c)
            b = self.read_from_source(f)
            assert a == b + c

class aubio_source_readmulti_test_case(aubio_source_read_test_case):

    def read_from_source(self, f):
        total_frames = 0
        while True:
            vec, read = f.do_multi()
            total_frames += read
            if read < f.hop_size: break
        print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
        print "(", total_frames, "frames", "in",
        print f.channels, "channels and",
        print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
        print "from", f.uri
        return total_frames

if __name__ == '__main__':
    from unittest import main
    main()

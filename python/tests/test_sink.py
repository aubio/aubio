#! /usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal
from aubio import fvec, source, sink
from numpy import array
from utils import list_all_sounds, get_tmp_sink_path, del_tmp_sink_path

list_of_sounds = list_all_sounds('sounds')
path = None

many_files = 300 # 256 opened files is too much

class aubio_sink_test_case(TestCase):

    def test_many_sinks(self):
        from tempfile import mkdtemp
        import os.path
        import shutil
        tmpdir = mkdtemp()
        sink_list = []
        for i in range(many_files):
            path = os.path.join(tmpdir, 'f-' + str(i) + '.wav')
            g = sink(path, 0)
            sink_list.append(g)
            write = 32
            for n in range(200):
                vec = fvec(write)
                g(vec, write)
            g.close()
        shutil.rmtree(tmpdir)

    def test_many_sinks_not_closed(self):
        from tempfile import mkdtemp
        import os.path
        import shutil
        tmpdir = mkdtemp()
        sink_list = []
        try:
            for i in range(many_files):
                path = os.path.join(tmpdir, 'f-' + str(i) + '.wav')
                g = sink(path, 0)
                sink_list.append(g)
                write = 256
                for n in range(200):
                    vec = fvec(write)
                    g(vec, write)
        except StandardError:
            pass
        else:
            self.fail("does not fail on too many files open")
        for g in sink_list:
            g.close()
        shutil.rmtree(tmpdir)

    def test_read_and_write(self):

        if not len(list_of_sounds):
            self.skipTest('add some sound files in \'python/tests/sounds\'')

        for path in list_of_sounds:
            for samplerate, hop_size in zip([0, 44100, 8000, 32000], [512, 1024, 64, 256]):
                f = source(path, samplerate, hop_size)
                if samplerate == 0: samplerate = f.samplerate
                sink_path = get_tmp_sink_path()
                g = sink(sink_path, samplerate)
                total_frames = 0
                while True:
                    vec, read = f()
                    g(vec, read)
                    total_frames += read
                    if read < f.hop_size: break
                if 0:
                    print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
                    print "(", total_frames, "frames", "in",
                    print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
                    print "from", f.uri,
                    print "to", g.uri
                del_tmp_sink_path(sink_path)

    def test_read_and_write_multi(self):

        if not len(list_of_sounds):
            self.skipTest('add some sound files in \'python/tests/sounds\'')

        for path in list_of_sounds:
            for samplerate, hop_size in zip([0, 44100, 8000, 32000], [512, 1024, 64, 256]):
                f = source(path, samplerate, hop_size)
                if samplerate == 0: samplerate = f.samplerate
                sink_path = get_tmp_sink_path()
                g = sink(sink_path, samplerate, channels = f.channels)
                total_frames = 0
                while True:
                    vec, read = f.do_multi()
                    g.do_multi(vec, read)
                    total_frames += read
                    if read < f.hop_size: break
                if 0:
                    print "read", "%.2fs" % (total_frames / float(f.samplerate) ),
                    print "(", total_frames, "frames", "in",
                    print f.channels, "channels", "in",
                    print total_frames / f.hop_size, "blocks", "at", "%dHz" % f.samplerate, ")",
                    print "from", f.uri,
                    print "to", g.uri,
                    print "in", g.channels, "channels"
                del_tmp_sink_path(sink_path)

    def test_close_file(self):
        samplerate = 44100
        sink_path = get_tmp_sink_path()
        g = sink(sink_path, samplerate)
        g.close()
        del_tmp_sink_path(sink_path)

    def test_close_file_twice(self):
        samplerate = 44100
        sink_path = get_tmp_sink_path()
        g = sink(sink_path, samplerate)
        g.close()
        g.close()
        del_tmp_sink_path(sink_path)

if __name__ == '__main__':
    from unittest import main
    main()

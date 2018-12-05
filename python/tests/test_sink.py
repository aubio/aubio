#! /usr/bin/env python

from nose2 import main
from nose2.tools import params
from numpy.testing import TestCase
from aubio import fvec, source, sink
from .utils import list_all_sounds, get_tmp_sink_path, del_tmp_sink_path

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

list_of_sounds = list_all_sounds('sounds')
samplerates = [0, 44100, 8000, 32000]
hop_sizes = [512, 1024, 64]

path = None

many_files = 300 # 256 opened files is too much

all_params = []
for soundfile in list_of_sounds:
    for hop_size in hop_sizes:
        for samplerate in samplerates:
            all_params.append((hop_size, samplerate, soundfile))

class aubio_sink_test_case(TestCase):

    def setUp(self):
        if not len(list_of_sounds):
            self.skipTest('add some sound files in \'python/tests/sounds\'')

    def test_wrong_filename(self):
        with self.assertRaises(RuntimeError):
            sink('')

    def test_wrong_samplerate(self):
        with self.assertRaises(RuntimeError):
            sink(get_tmp_sink_path(), -1)

    def test_wrong_samplerate_too_large(self):
        with self.assertRaises(RuntimeError):
            sink(get_tmp_sink_path(), 1536001, 2)

    def test_wrong_channels(self):
        with self.assertRaises(RuntimeError):
            sink(get_tmp_sink_path(), 44100, -1)

    def test_wrong_channels_too_large(self):
        with self.assertRaises(RuntimeError):
            sink(get_tmp_sink_path(), 44100, 202020)

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
            for _ in range(200):
                vec = fvec(write)
                g(vec, write)
            g.close()
        shutil.rmtree(tmpdir)

    @params(*all_params)
    def test_read_and_write(self, hop_size, samplerate, path):

        try:
            f = source(path, samplerate, hop_size)
        except RuntimeError as e:
            self.skipTest('failed opening with hop_s = {:d}, samplerate = {:d} ({:s})'.format(hop_size, samplerate, str(e)))
        if samplerate == 0: samplerate = f.samplerate
        sink_path = get_tmp_sink_path()
        g = sink(sink_path, samplerate)
        total_frames = 0
        while True:
            vec, read = f()
            g(vec, read)
            total_frames += read
            if read < f.hop_size: break
        del_tmp_sink_path(sink_path)

    @params(*all_params)
    def test_read_and_write_multi(self, hop_size, samplerate, path):
        try:
            f = source(path, samplerate, hop_size)
        except RuntimeError as e:
            self.skipTest('failed opening with hop_s = {:d}, samplerate = {:d} ({:s})'.format(hop_size, samplerate, str(e)))
        if samplerate == 0: samplerate = f.samplerate
        sink_path = get_tmp_sink_path()
        g = sink(sink_path, samplerate, channels = f.channels)
        total_frames = 0
        while True:
            vec, read = f.do_multi()
            g.do_multi(vec, read)
            total_frames += read
            if read < f.hop_size: break
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

    def test_read_with(self):
        samplerate = 44100
        sink_path = get_tmp_sink_path()
        vec = fvec(128)
        with sink(sink_path, samplerate) as g:
            for _ in range(10):
                g(vec, 128)

if __name__ == '__main__':
    main()

#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

Compare the speed of several methods for reading and loading a sound file.

This file depends on audioread and librosa:
    https://github.com/beetbox/audioread
    https://github.com/bmcfee/librosa

"""

import numpy as np
import aubio
import audioread
import librosa

def read_file_audioread(filename):
    # taken from librosa.util.utils
    def convert_buffer_to_float(buf, n_bytes = 2, dtype = np.float32):
        # Invert the scale of the data
        scale = 1./float(1 << ((8 * n_bytes) - 1))
        # Construct the format string
        fmt = '<i{:d}'.format(n_bytes)
        # Rescale and format the data buffer
        out = scale * np.frombuffer(buf, fmt).astype(dtype)
        out = out.reshape(2, -1)
        return out

    with audioread.audio_open(filename) as f:
        total_frames = 0
        for buf in f:
            samples = convert_buffer_to_float(buf)
            total_frames += samples.shape[1]
        return total_frames, f.samplerate

def load_file_librosa(filename):
    y, sr = librosa.load(filename, sr = None)
    return len(y), sr

def read_file_aubio(filename):
    f = aubio.source(filename, hop_size = 1024)
    total_frames = 0
    while True:
        samples, read = f()
        total_frames += read
        if read < f.hop_size: break
    return total_frames, f.samplerate

def load_file_aubio(filename):
    f = aubio.source(filename, hop_size = 1024)
    y = np.zeros(f.duration, dtype = aubio.float_type)
    total_frames = 0
    while True:
        samples, read = f()
        y[total_frames:total_frames + read] = samples[:read]
        total_frames += read
        if read < f.hop_size: break
    assert len(y) == total_frames
    return total_frames, f.samplerate

def test_speed(function, filename):
    times = []
    for i in range(10):
        start = time.time()
        total_frames, samplerate = function(filename)
        elapsed = time.time() - start
        #print ("{:5f} ".format(elapsed)),
        times.append(elapsed)
    #print
    times = np.array(times)
    duration_min = int(total_frames/float(samplerate) // 60)
    str_format = '{:25s} took {:5f} seconds avg (±{:5f}) to run on a ~ {:d} minutes long file'
    print (str_format.format(function.__name__, times.mean(), times.std(), duration_min ))

if __name__ == '__main__':
    import sys, time
    if len(sys.argv) < 2:
        print ("not enough arguments")
        sys.exit(1)
    filename = sys.argv[1]

    functions = [read_file_aubio, load_file_aubio, read_file_audioread, load_file_librosa]
    for f in functions:
        test_speed(f, filename)

#! /usr/bin/env python

""" Create a simple stereo file containing a sine tone at 441 Hz, using only
numpy and python's native wave module. """

import wave
import numpy as np


def create_sine_wave(freq, samplerate, nframes, nchannels):
    """ create a pure tone """
    # samples indices
    _t = np.tile(np.arange(nframes), (nchannels, 1))
    # sine wave generation
    _x = 0.7 * np.sin(2. * np.pi * freq * _t / float(samplerate))
    # conversion to int and channel interleaving
    return (_x * 32767.).astype(np.int16).T.flatten()


def create_test_sound(pathname, freq=441, duration=None,
                      sampwidth=2, framerate=44100, nchannels=2):
    """ create a sound file at pathname, overwriting exiting file """
    nframes = duration or framerate  # defaults to 1 second duration
    fid = wave.open(pathname, 'w')
    fid.setnchannels(nchannels)
    fid.setsampwidth(sampwidth)
    fid.setframerate(framerate)
    fid.setnframes(nframes)
    frames = create_sine_wave(freq, framerate, nframes, nchannels)
    fid.writeframes(frames.tobytes())
    fid.close()
    return 0


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        sys.exit(2)
    sys.exit(create_test_sound(sys.argv[1]))

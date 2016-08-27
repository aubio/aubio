#! /usr/bin/env python
import sys, aubio

samplerate = 0  # use original source samplerate
hop_size = 256 # number of frames to read in one block
s = aubio.source(sys.argv[1], samplerate, hop_size)
total_frames = 0

while True: # reading loop
    samples, read = s()
    total_frames += read
    if read < hop_size: break # end of file reached

fmt_string = "read {:d} frames at {:d}Hz from {:s}"
print (fmt_string.format(total_frames, s.samplerate, sys.argv[1]))


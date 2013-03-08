#! /usr/bin/env python

import sys
from aubio import source, pitch

win_s = 1024 # fft size
hop_s = win_s # hop size

if len(sys.argv) < 2:
    print "Usage: %s <filename> [samplerate]" % sys.argv[0]
    sys.exit(1)

filename = sys.argv[1]

samplerate = 0
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

s = source(filename, samplerate, hop_s)
samplerate = s.samplerate

pitch_o = pitch("default", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")

pitches = []


# total number of frames read
total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    print "%f %f" % (total_frames / float(samplerate), pitch)
    #pitches += [pitches]
    total_frames += read
    if read < hop_s: break

#print pitches

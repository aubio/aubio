#! /usr/bin/env python

import sys
from aubio import onset, source
from numpy import array, hstack, zeros

win_s = 512                 # fft size
hop_s = win_s / 2           # hop size

if len(sys.argv) < 2:
    print "Usage: %s <filename> [samplerate]" % sys.argv[0]
    sys.exit(1)

filename = sys.argv[1]

samplerate = 0
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

s = source(filename, samplerate, hop_s)
samplerate = s.samplerate
o = onset("default", win_s, hop_s)

# onset detection delay, in blocks
delay = 4. * hop_s

onsets = []
total_frames = 0
while True:
    samples, read = s()
    is_onset = o(samples)
    if is_onset:
        this_onset = int(total_frames - delay + is_onset[0] * hop_s)
        print "%f" % (this_onset / float(samplerate))
        onsets.append(this_onset)
    total_frames += read
    if read < hop_s: break
#print len(onsets)

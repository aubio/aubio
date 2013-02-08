#! /usr/bin/env python

import sys
from aubio import tempo, source

win_s = 512                 # fft size
hop_s = win_s / 2           # hop size
samplerate = 44100

if len(sys.argv) < 2:
    print "Usage: %s <filename>" % sys.argv[0]
    sys.exit(1)

filename = sys.argv[1]
beats = []

s = source(filename, samplerate, hop_s)
t = tempo("default", win_s, hop_s)

block_read = 0
while True:
    samples, read = s()
    isbeat = t(samples)
    if isbeat:
        thisbeat = (block_read * hop_s + isbeat[0]) / samplerate
        print "%.4f" % thisbeat
        beats.append (thisbeat)
    block_read += 1
    if read < hop_s: break

periods = [60./(b - a) for a,b in zip(beats[:-1],beats[1:])]

from numpy import mean, median
print 'mean period:', mean(periods), 'bpm'
print 'median period:', median(periods), 'bpm'

from pylab import plot, show
plot(beats[1:], periods)
show()

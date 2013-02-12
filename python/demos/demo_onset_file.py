#! /usr/bin/env python

import sys
from aubio import onset, source
from numpy import array, hstack, zeros

win_s = 512                 # fft size
hop_s = win_s / 2           # hop size
samplerate = 44100
downsample = 2              # used to plot n samples / hop_s

if len(sys.argv) < 2:
    print "Usage: %s <filename>" % sys.argv[0]
    sys.exit(1)

filename = sys.argv[1]
onsets = []
oldonsets = []

s = source(filename, samplerate, hop_s)
o = onset("default", win_s, hop_s)

block_read = 0
allsamples_max = zeros(0,)
while True:
    samples, read = s()
    new_maxes = (abs(samples.reshape(hop_s/downsample, downsample))).max(axis=0)
    allsamples_max = hstack([allsamples_max, new_maxes])
    isbeat = o(samples)
    if isbeat:
        thisbeat = (block_read - 4. + isbeat[0]) * hop_s / samplerate
        print "%.4f" % thisbeat
        onsets.append (thisbeat)
        # old onset
        thisbeat = (block_read - 3. ) * hop_s / samplerate
        oldonsets.append (thisbeat)
    block_read += 1
    if read < hop_s: break

# do plotting
from numpy import arange
from pylab import plot, show, xlabel, ylabel, legend, ylim, subplot, axis
allsamples_max = (allsamples_max > 0) * allsamples_max
allsamples_max_times = [ float(t) * hop_s / downsample / samplerate for t in range(len(allsamples_max)) ]
plot(allsamples_max_times,  allsamples_max, '-b')
plot(allsamples_max_times, -allsamples_max, '-b')
axis(xmin = 0., xmax = max(allsamples_max_times) )
for stamp in onsets: plot([stamp, stamp], [-1., 1.], '.-r')
for stamp in oldonsets: plot([stamp, stamp], [-1., 1.], '.-g')
xlabel('time (s)')
ylabel('amplitude')
show()


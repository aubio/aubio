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

s = source(filename, samplerate, hop_s)
o = onset("default", win_s, hop_s)

desc = []
tdesc = []

block_read = 0
allsamples_max = zeros(0,)
while True:
    samples, read = s()
    new_maxes = (abs(samples.reshape(hop_s/downsample, downsample))).max(axis=0)
    allsamples_max = hstack([allsamples_max, new_maxes])
    isbeat = o(samples)
    desc.append(o.get_descriptor())
    tdesc.append(o.get_thresholded_descriptor())
    if isbeat:
        thisbeat = (block_read - 4. + isbeat[0]) * hop_s / samplerate
        print "%.4f" % thisbeat
        onsets.append (thisbeat)
    block_read += 1
    if read < hop_s: break

# do plotting
from numpy import arange
import matplotlib.pyplot as plt
allsamples_max = (allsamples_max > 0) * allsamples_max
allsamples_max_times = [ float(t) * hop_s / downsample / samplerate for t in range(len(allsamples_max)) ]
plt1 = plt.axes([0.1, 0.75, 0.8, 0.19])
plt2 = plt.axes([0.1, 0.1, 0.8, 0.65], sharex = plt1)
plt.rc('lines',linewidth='.8')
plt1.plot(allsamples_max_times,  allsamples_max, '-b')
plt1.plot(allsamples_max_times, -allsamples_max, '-b')
for stamp in onsets: plt1.plot([stamp, stamp], [-1., 1.], '-r')
plt1.axis(xmin = 0., xmax = max(allsamples_max_times) )
plt1.xaxis.set_visible(False)
desc_times = [ float(t) * hop_s / samplerate for t in range(len(desc)) ]
desc_plot = [d / max(desc) for d in desc]
plt2.plot(desc_times, desc_plot, '-g')
tdesc_plot = [d / max(desc) for d in tdesc]
for stamp in onsets: plt2.plot([stamp, stamp], [min(tdesc_plot), max(desc_plot)], '-r')
plt2.plot(desc_times, tdesc_plot, '-y')
plt2.axis(ymin = min(tdesc_plot), ymax = max(desc_plot))
plt.xlabel('time (s)')
#plt.savefig('/tmp/t.png', dpi=200)
plt.show()

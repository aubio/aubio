#! /usr/bin/env python

import sys
from aubio import pvoc, source
from numpy import zeros, hstack

def get_waveform_plot(filename, samplerate = 0, ax = None):
    import matplotlib.pyplot as plt
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    hop_s = 512                                        # fft window size

    allsamples_max = zeros(0,)
    downsample = 2  # to plot n samples / hop_s

    a = source(filename, samplerate, hop_s)            # source file
    if samplerate == 0: samplerate = a.samplerate

    total_frames = 0
    while True:
        samples, read = a()
        # keep some data to plot it later
        new_maxes = (abs(samples.reshape(hop_s/downsample, downsample))).max(axis=0)
        allsamples_max = hstack([allsamples_max, new_maxes])
        total_frames += read
        if read < hop_s: break

    allsamples_max = (allsamples_max > 0) * allsamples_max
    allsamples_max_times = [ ( float (t) / downsample ) * hop_s for t in range(len(allsamples_max)) ]

    ax.plot(allsamples_max_times,  allsamples_max, '-b')
    ax.plot(allsamples_max_times, -allsamples_max, '-b')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: %s <filename>" % sys.argv[0]
    else:
        for soundfile in sys.argv[1:]:
            get_waveform_plot(soundfile)
            # display graph
            show()

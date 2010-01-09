#! /usr/bin/python

import sys
from os.path import splitext, basename
from aubio import tempo 
from aubioinput import aubioinput

win_s = 512                 # fft size
hop_s = win_s / 2           # hop size
beat = tempo("default", win_s, hop_s)

beats = []

def process(samples, pos):
    isbeat = beat(samples)
    if isbeat:
        thisbeat = (float(isbeat[0]) + pos * hop_s) / 44100.
        print thisbeat
        beats.append (thisbeat)

if len(sys.argv) < 2:
    print "Usage: %s <filename>" % sys.argv[0]
else:
    filename = sys.argv[1]
    a = aubioinput(filename, process = process, hopsize = hop_s,
            caps = 'audio/x-raw-float, rate=44100, channels=1')
    a.run()
    print beats 

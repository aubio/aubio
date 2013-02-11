#! /usr/bin/env python

import sys
from math import sin, pi
from aubio import sink
from numpy import array

if len(sys.argv) != 2:
    print 'usage: %s <outputfile>' % sys.argv[0]
    sys.exit(1)

samplerate = 44100      # in Hz
pitch = 2200            # in Hz
blocksize = 256         # in samples
duration = 0.02         # in seconds

twopi = pi * 2.

duration = int ( 44100 * duration ) # convert to samples
attack = 3

period = int ( float(samplerate) /  pitch )
sinetone = [ 0.7 * sin(twopi * i/ period) for i in range(period) ] 
sinetone *= int ( duration / period )
sinetone = array(sinetone, dtype = 'float32')

from math import exp, e
for i in range(len(sinetone)):
    sinetone[i] *= exp( - e * float(i) / len(sinetone))
for i in range(attack):
    sinetone[i] *= exp( e * (float(i) / attack - 1 ) )

my_sink = sink(sys.argv[1], 44100)

i = 0
while i + blocksize < duration:
    my_sink(sinetone[i:i+blocksize], blocksize)
    i += blocksize

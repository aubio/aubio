#!/usr/bin/env python

import pylab
import numpy
import sys

from aubio.aubiowrapper import * 

win_size = 2048 
channels = 1
n_filters = 40
samplerate = 44100

filterbank = new_aubio_filterbank_mfcc(n_filters, win_size, samplerate, 
        0., samplerate) 


mfcc_filters = []
for channel in range(n_filters):
  vec = aubio_filterbank_getchannel(filterbank,channel)
  mfcc_filters.append([])
  for index in range(win_size): 
    mfcc_filters[channel].append(fvec_read_sample(vec,0,index))

doLog=False
if len(sys.argv)>1: 
  if sys.argv[1]=='log':
    doLog=True

nmat= numpy.array(mfcc_filters)

pylab.hold(True)

n_filters=numpy.shape(nmat)[0]
for i in range(n_filters):
  if doLog==True:
    pylab.semilogx(nmat[i,:])
  else:
    pylab.plot(nmat[i,:]) 

pylab.hold(False)
#pylab.savefig('test.png')
pylab.show()

#!/usr/bin/env python

import pylab
import numpy
import sys


filename=sys.argv[1]

doLog=False
if len(sys.argv)>2: 
  if sys.argv[2]=='log':
    doLog=True

mat = pylab.load(filename)
nmat= numpy.array(mat)
print numpy.shape(nmat)

pylab.hold(True)

n_filters=numpy.shape(nmat)[0]
for i in range(n_filters):
  if doLog==True:
    pylab.semilogx(nmat[i,:])
  else:
    pylab.plot(nmat[i,:]) 


pylab.hold(False)
pylab.show()
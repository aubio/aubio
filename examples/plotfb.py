#!/usr/bin/env python

import pylab
import numpy
import sys

filename=sys.argv[1]

mat = pylab.load(filename)
nmat= numpy.array(mat)
print numpy.shape(nmat)

pylab.hold(True)

n_filters=numpy.shape(nmat)[0]
for i in range(n_filters):
  pylab.plot(nmat[i,:])


pylab.hold(False)
pylab.show()
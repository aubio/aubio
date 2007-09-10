#!/usr/bin/env python

import pylab
import numpy
import sys

filename=sys.argv[1]

mat=pylab.load(filename)
nmat=numpy.array(mat).T
print numpy.shape(nmat)


pylab.matshow(nmat, cmap=pylab.cm.gray, aspect='auto')
#pylab.imshow(nmat, cmap=pylab.cm.gray, aspect='auto', interpolation=False)
#pylab.contour(nmat, cmap=pylab.cm.gray, aspect='auto')

pylab.show()
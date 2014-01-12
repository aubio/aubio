#! /usr/bin/env python

import numpy
from _aubio import *
from midiconv import *
from slicing import *

class fvec(numpy.ndarray):
    " a simple numpy array holding a vector of float32 "
    def __new__(self, length = 1024, **kwargs):
        self.length = length
        if type(length) == type([]):
            return numpy.array(length, dtype='float32', **kwargs)
        return numpy.zeros(length, dtype='float32', **kwargs)

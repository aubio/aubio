import numpy
from _aubio import *

class fvec(numpy.ndarray):

    def __new__(self, length = 1024, **kwargs):
        if type(length) == type([]):
            return numpy.array(length, dtype='float32', **kwargs)
        return numpy.zeros(length, dtype='float32', **kwargs)

import numpy
from _aubio import *

class fvec(numpy.ndarray):

    def __init__(self, length = 1024, **kwargs):
        super(numpy.ndarray, self).__init__(**kwargs)

    def __new__(self, length = 1024, **kwargs):
        self = numpy.zeros(length, dtype='float32', **kwargs)
        return self

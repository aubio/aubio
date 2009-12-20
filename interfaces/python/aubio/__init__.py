import numpy

class fvec(numpy.ndarray):

    def __init__(self, length = 1024, **kwargs):
        super(numpy.ndarray, self).__init__(**kwargs)

    def __new__(self, length = 1024, **kwargs):
        self = numpy.zeros(length, dtype='float32', **kwargs)
        return self

class cvec:

    def __init__ (self, length = 1024, **kwargs):
        self.norm = numpy.zeros(length / 2 + 1, dtype='float32', **kwargs)
        self.phas = numpy.zeros(length / 2 + 1, dtype='float32', **kwargs)

    def __len__ (self):
        assert len(self.norm) == len(self.phas)
        return len(self.norm)

#! /usr/bin/env python

import numpy
from ._aubio import *
from .midiconv import *
from .slicing import *

class fvec(numpy.ndarray):
    """a numpy vector holding audio samples"""

    def __new__(cls, input_arg=1024, **kwargs):
        if isinstance(input_arg, int):
            return numpy.zeros(input_arg, dtype=float_type, **kwargs)
        else:
            return numpy.array(input_arg, dtype=float_type, **kwargs)

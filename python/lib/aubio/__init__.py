#! /usr/bin/env python

import numpy
from ._aubio import __version__ as version
from ._aubio import float_type
from ._aubio import *
from .midiconv import *
from .slicing import *

class fvec(numpy.ndarray):
    """a numpy vector holding audio samples"""

    def __new__(cls, input_arg=1024, **kwargs):
        if isinstance(input_arg, int):
            if input_arg == 0:
                raise ValueError("vector length of 1 or more expected")
            return numpy.zeros(input_arg, dtype=float_type, **kwargs)
        else:
            return numpy.array(input_arg, dtype=float_type, **kwargs)

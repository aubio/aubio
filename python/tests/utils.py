#! /usr/bin/env python

def array_from_text_file(filename, dtype = 'float'):
    import os.path
    from numpy import array
    filename = os.path.join(os.path.dirname(__file__), filename)
    return array([line.split() for line in open(filename).readlines()],
        dtype = dtype)

def list_all_sounds(rel_dir):
    import os.path, glob
    datadir = os.path.join(os.path.dirname(__file__), rel_dir)
    return glob.glob(os.path.join(datadir,'*.*'))

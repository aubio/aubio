#! /usr/bin/python

from distutils.core import setup, Extension

from generator import generate_object_files

setup(name="_aubio", version="1.0",
      packages = ['aubio'],
      ext_modules = [ 
        Extension("_aubio",
            ["aubiomodule.c",
            "aubioproxy.c",
            "py-cvec.c",
            # example without macro
            "py-filter.c",
            # macroised
            "py-filterbank.c",
            "py-fft.c",
            "py-phasevoc.c",
            # generated files
            ] + generate_object_files(),
            include_dirs=['../../build/src', '../../src', '.' ],
            library_dirs=['../../build/src', '../../src/.libs' ],
            libraries=['aubio'])])


#! /usr/bin/python

from distutils.core import setup, Extension

from generator import generate_object_files

import os.path

library_dirs = ['../../build/src', '../../src/.libs']
include_dirs = ['../../build/src', '../../src', '.' ]
library_dirs = filter (lambda x: os.path.isdir(x), library_dirs)
include_dirs = filter (lambda x: os.path.isdir(x), include_dirs)

aubio_extension = Extension("_aubio",
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
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            libraries=['aubio'])

setup(name='aubio',
      version = '0.4.0alpha',
      packages = ['aubio'],
      description = 'interface to the aubio library',
      long_description = 'interface to the aubio library',
      license = 'GNU/GPL version 3',
      author = 'Paul Brossier',
      author_email = 'piem@aubio.org',
      maintainer = 'Paul Brossier',
      maintainer_email = 'piem@aubio.org',
      url = 'http://aubio.org/',
      ext_modules = [aubio_extension])


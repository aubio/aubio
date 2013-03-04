#! /usr/bin/python

from distutils.core import setup, Extension
from generator import generate_object_files
import sys
import os.path
import numpy

# read from VERSION
for l in open(os.path.join('..','VERSION')).readlines(): exec (l.strip())
__version__ = '.'.join \
        ([str(x) for x in [AUBIO_MAJOR_VERSION, AUBIO_MINOR_VERSION, AUBIO_PATCH_VERSION]]) \
        + AUBIO_VERSION_STATUS

library_dirs = ['../build/src', '../src/.libs']
include_dirs = ['../build/src', '../src', '.' ]
library_dirs = filter (lambda x: os.path.isdir(x), library_dirs)
include_dirs = filter (lambda x: os.path.isdir(x), include_dirs)

aubio_extension = Extension("aubio._aubio", [
            "ext/aubiomodule.c",
            "ext/aubioproxy.c",
            "ext/ufuncs.c",
            "ext/py-cvec.c",
            # example without macro
            "ext/py-filter.c",
            # macroised
            "ext/py-filterbank.c",
            "ext/py-fft.c",
            "ext/py-phasevoc.c",
            # generated files
            ] + generate_object_files(),
        include_dirs = include_dirs + [ numpy.get_include() ],
        library_dirs = library_dirs,
        libraries=['aubio'])

if sys.platform.startswith('darwin'):
        aubio_extension.extra_link_args = ['-framework','CoreFoundation', '-framework','AudioToolbox']

classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Operating System :: POSIX',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        ]

distrib = setup(name='aubio',
        version = __version__,
        packages = ['aubio'],
        ext_modules = [aubio_extension],
        description = 'interface to the aubio library',
        long_description = 'interface to the aubio library',
        license = 'GNU/GPL version 3',
        author = 'Paul Brossier',
        author_email = 'piem@aubio.org',
        maintainer = 'Paul Brossier',
        maintainer_email = 'piem@aubio.org',
        url = 'http://aubio.org/',
        platforms = 'any',
        classifiers = classifiers,
        )


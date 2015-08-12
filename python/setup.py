#! /usr/bin/env python

from setuptools import setup, Extension

import sys
import os.path
import numpy

# read from VERSION
for l in open('VERSION').readlines(): exec (l.strip())
__version__ = '.'.join \
        ([str(x) for x in [AUBIO_MAJOR_VERSION, AUBIO_MINOR_VERSION, AUBIO_PATCH_VERSION]]) \
        + AUBIO_VERSION_STATUS


include_dirs = []
library_dirs = []
define_macros = []
extra_link_args = []

include_dirs += ['ext']
include_dirs += [ numpy.get_include() ]

if sys.platform.startswith('darwin'):
    extra_link_args += ['-framework','CoreFoundation', '-framework','AudioToolbox']

output_path = 'gen'
generated_object_files = []

if not os.path.isdir(output_path):
    from lib.generator import generate_object_files
    generated_object_files = generate_object_files(output_path)
    # define include dirs
else:
    import glob
    generated_object_files = glob.glob(os.path.join(output_path, '*.c'))
include_dirs += [output_path]

if os.path.isfile('../src/aubio.h'):
    define_macros += [('USE_LOCAL_AUBIO', 1)]
    include_dirs += ['../src'] # aubio.h
    include_dirs += ['../build/src'] # config.h
    library_dirs += ['../build/src']

aubio_extension = Extension("aubio._aubio", [
    "ext/aubiomodule.c",
    "ext/aubioproxy.c",
    "ext/ufuncs.c",
    "ext/py-musicutils.c",
    "ext/py-cvec.c",
    # example without macro
    "ext/py-filter.c",
    # macroised
    "ext/py-filterbank.c",
    "ext/py-fft.c",
    "ext/py-phasevoc.c",
    "ext/py-source.c",
    "ext/py-sink.c",
    # generated files
    ] + generated_object_files,
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    extra_link_args = extra_link_args,
    define_macros = define_macros,
    libraries=['aubio'])

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
    package_dir = {'aubio':'lib/aubio'},
    scripts = ['scripts/aubiocut'],
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
    install_requires = ['numpy'],
    )

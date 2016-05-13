#! /usr/bin/env python

import sys
import os.path
import numpy
from setuptools import setup, Extension
from python.lib.moresetuptools import CleanGenerated, GenerateCommand
# function to generate gen/*.{c,h}
from python.lib.gen_external import generate_external, header, output_path

# read from VERSION
for l in open('VERSION').readlines(): exec (l.strip())
__version__ = '.'.join \
        ([str(x) for x in [AUBIO_MAJOR_VERSION, AUBIO_MINOR_VERSION, AUBIO_PATCH_VERSION]]) \
        + AUBIO_VERSION_STATUS

include_dirs = []
library_dirs = []
define_macros = []
extra_link_args = []

include_dirs += [ 'python/ext' ]
include_dirs += [ output_path ] # aubio-generated.h
include_dirs += [ numpy.get_include() ]

if sys.platform.startswith('darwin'):
    extra_link_args += ['-framework','CoreFoundation', '-framework','AudioToolbox']

if os.path.isfile('src/aubio.h'):
    define_macros += [('USE_LOCAL_AUBIO', 1)]
    include_dirs += ['src'] # aubio.h
    library_dirs += ['build/src']

aubio_extension = Extension("aubio._aubio", [
    "python/ext/aubiomodule.c",
    "python/ext/aubioproxy.c",
    "python/ext/ufuncs.c",
    "python/ext/py-musicutils.c",
    "python/ext/py-cvec.c",
    "python/ext/py-filter.c",
    "python/ext/py-filterbank.c",
    "python/ext/py-fft.c",
    "python/ext/py-phasevoc.c",
    "python/ext/py-source.c",
    "python/ext/py-sink.c",
    # generate files if they don't exit
    ] + generate_external(header, output_path, overwrite = False),
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
    package_dir = {'aubio':'python/lib/aubio'},
    scripts = ['python/scripts/aubiocut'],
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
    cmdclass = {
        'clean': CleanGenerated,
        'generate': GenerateCommand,
        },
    test_suite = 'nose2.collector.collector',
    )

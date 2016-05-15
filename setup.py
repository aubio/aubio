#! /usr/bin/env python

import sys, os.path, glob
from setuptools import setup, Extension
from python.lib.moresetuptools import *
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
try:
    import numpy
    include_dirs += [ numpy.get_include() ]
except ImportError:
    pass

if sys.platform.startswith('darwin'):
    extra_link_args += ['-framework','CoreFoundation', '-framework','AudioToolbox']

sources = glob.glob(os.path.join('python', 'ext', '*.c'))

aubio_extension = Extension("aubio._aubio",
    sources,
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    extra_link_args = extra_link_args,
    define_macros = define_macros)

if os.path.isfile('src/aubio.h'):
    # if aubio headers are found in this directory
    add_local_aubio_header(aubio_extension)
    # was waf used to build the shared lib?
    if os.path.isdir(os.path.join('build','src')):
        # link against build/src/libaubio, built with waf
        add_local_aubio_lib(aubio_extension)
    else:
        # add libaubio sources and look for optional deps with pkg-config
        add_local_aubio_sources(aubio_extension)
        __version__ += '_libaubio'
else:
    # look for aubio headers and lib using pkg-config
    add_system_aubio(aubio_extension)


# generate files if they don't exit
aubio_extension.sources += generate_external(header, output_path, overwrite = False)

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

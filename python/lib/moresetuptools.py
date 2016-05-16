""" A collection of function used from setup.py distutils script """
#
import sys, os, glob, subprocess
import distutils, distutils.command.clean, distutils.dir_util
from .gen_external import generate_external, header, output_path

# inspired from https://gist.github.com/abergmeier/9488990
def add_packages(packages, ext=None, **kw):
    """ use pkg-config to search which of 'packages' are installed """
    flag_map = {
        '-I': 'include_dirs',
        '-L': 'library_dirs',
        '-l': 'libraries'}

    # if a setuptools extension is passed, fill it with pkg-config results
    if ext:
        kw = {'include_dirs': ext.include_dirs,
              'extra_link_args': ext.extra_link_args,
              'library_dirs': ext.library_dirs,
              'libraries': ext.libraries,
             }

    for package in packages:
        cmd = ['pkg-config', '--libs', '--cflags', package]
        try:
            tokens = subprocess.check_output(cmd)
        except Exception as e:
            print("Running \"{:s}\" failed: {:s}".format(' '.join(cmd), repr(e)))
            continue
        tokens = tokens.decode('utf8').split()
        for token in tokens:
            key = token[:2]
            try:
                arg = flag_map[key]
                value = token[2:]
            except KeyError:
                arg = 'extra_link_args'
                value = token
            kw.setdefault(arg, []).append(value)
    for key, value in iter(kw.items()): # remove duplicated
        kw[key] = list(set(value))
    return kw

def add_local_aubio_header(ext):
    """ use local "src/aubio.h", not <aubio/aubio.h>"""
    ext.define_macros += [('USE_LOCAL_AUBIO', 1)]
    ext.include_dirs += ['src'] # aubio.h

def add_local_aubio_lib(ext):
    """ add locally built libaubio from build/src """
    print("Info: using locally built libaubio")
    ext.library_dirs += [os.path.join('build', 'src')]
    ext.libraries += ['aubio']

def add_local_aubio_sources(ext):
    """ build aubio inside python module instead of linking against libaubio """
    print("Warning: libaubio was not built with waf, adding src/")
    # create an empty header, macros will be passed on the command line
    fake_config_header = os.path.join('python', 'ext', 'config.h')
    distutils.file_util.write_file(fake_config_header, "")
    aubio_sources = glob.glob(os.path.join('src', '**.c'))
    aubio_sources += glob.glob(os.path.join('src', '*', '**.c'))
    ext.sources += aubio_sources
    # define macros (waf puts them in build/src/config.h)
    for define_macro in ['HAVE_STDLIB_H', 'HAVE_STDIO_H',
                         'HAVE_MATH_H', 'HAVE_STRING_H',
                         'HAVE_C99_VARARGS_MACROS',
                         'HAVE_LIMITS_H', 'HAVE_MEMCPY_HACKS']:
        ext.define_macros += [(define_macro, 1)]

    # loof for additional packages
    print("Info: looking for *optional* additional packages")
    packages = ['libavcodec', 'libavformat', 'libavutil', 'libavresample',
                'jack',
                'sndfile', 'samplerate',
                #'fftw3f',
               ]
    add_packages(packages, ext=ext)
    if 'avcodec' in ext.libraries \
            and 'avformat' in ext.libraries \
            and 'avutil' in ext.libraries \
            and 'avresample' in ext.libraries:
        ext.define_macros += [('HAVE_LIBAV', 1)]
    if 'jack' in ext.libraries:
        ext.define_macros += [('HAVE_JACK', 1)]
    if 'sndfile' in ext.libraries:
        ext.define_macros += [('HAVE_SNDFILE', 1)]
    if 'samplerate' in ext.libraries:
        ext.define_macros += [('HAVE_SAMPLERATE', 1)]
    if 'fftw3f' in ext.libraries:
        ext.define_macros += [('HAVE_FFTW3F', 1)]
        ext.define_macros += [('HAVE_FFTW3', 1)]

    # add accelerate on darwin
    if sys.platform.startswith('darwin'):
        ext.extra_link_args += ['-framework', 'Accelerate']
        ext.define_macros += [('HAVE_ACCELERATE', 1)]
        ext.define_macros += [('HAVE_SOURCE_APPLE_AUDIO', 1)]
        ext.define_macros += [('HAVE_SINK_APPLE_AUDIO', 1)]

    if sys.platform.startswith('win'):
        ext.define_macros += [('HAVE_WIN_HACKS', 1)]

    ext.define_macros += [('HAVE_WAVWRITE', 1)]
    ext.define_macros += [('HAVE_WAVREAD', 1)]
    # TODO:
    # add cblas
    if 0:
        ext.libraries += ['cblas']
        ext.define_macros += [('HAVE_ATLAS_CBLAS_H', 1)]

def add_system_aubio(ext):
    # use pkg-config to find aubio's location
    add_packages(['aubio'], ext)
    if 'aubio' not in ext.libraries:
        print("Error: libaubio not found")

class CleanGenerated(distutils.command.clean.clean):
    def run(self):
        distutils.dir_util.remove_tree(output_path)
        distutils.command.clean.clean.run(self)

class GenerateCommand(distutils.cmd.Command):
    description = 'generate gen/gen-*.c files from ../src/aubio.h'
    user_options = [
            # The format is (long option, short option, description).
            ('enable-double', None, 'use HAVE_AUBIO_DOUBLE=1 (default: 0)'),
            ]

    def initialize_options(self):
        self.enable_double = False

    def finalize_options(self):
        if self.enable_double:
            self.announce(
                    'will generate code for aubio compiled with HAVE_AUBIO_DOUBLE=1',
                    level=distutils.log.INFO)

    def run(self):
        self.announce( 'Generating code', level=distutils.log.INFO)
        generated_object_files = generate_external(header, output_path, usedouble=self.enable_double)

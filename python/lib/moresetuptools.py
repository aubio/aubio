""" A collection of function used from setup.py distutils script """
#
import sys, os, glob, subprocess
import distutils, distutils.command.clean, distutils.dir_util
from .gen_external import generate_external, header, output_path

def get_aubio_version():
    # read from VERSION
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(this_file_dir, '..', '..', 'VERSION')

    if not os.path.isfile(version_file):
        raise SystemError("VERSION file not found.")

    for l in open(version_file).readlines():
        #exec (l.strip())
        if l.startswith('AUBIO_MAJOR_VERSION'):
            AUBIO_MAJOR_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_MINOR_VERSION'):
            AUBIO_MINOR_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_PATCH_VERSION'):
            AUBIO_PATCH_VERSION = int(l.split('=')[1])
        if l.startswith('AUBIO_VERSION_STATUS'):
            AUBIO_VERSION_STATUS = l.split('=')[1].strip()[1:-1]

    if AUBIO_MAJOR_VERSION is None or AUBIO_MINOR_VERSION is None \
            or AUBIO_PATCH_VERSION is None:
        raise SystemError("Failed parsing VERSION file.")

    verstr = '.'.join(map(str, [AUBIO_MAJOR_VERSION,
                                     AUBIO_MINOR_VERSION,
                                     AUBIO_PATCH_VERSION]))

    AUBIO_GIT_SHA = get_git_revision_hash()
    """ append sha to version in alpha release
    """
    if '~alpha' in AUBIO_VERSION_STATUS :
        if AUBIO_GIT_SHA:
            AUBIO_VERSION_STATUS = '~git'+AUBIO_GIT_SHA
    if AUBIO_VERSION_STATUS is not None :
        verstr += AUBIO_VERSION_STATUS
    return verstr

def get_aubio_pyversion():
    # convert to version for python according to pep 440
    # see https://www.python.org/dev/peps/pep-0440/
    verstr = get_aubio_version()
    if '~alpha' in verstr or '~git' in verstr:
        verstr = verstr.split('~')[0] + '+a1'
        gitsha = get_git_revision_hash(short=True)
        if gitsha:
            verstr+='.git.'+gitsha
    # TODO: add rc, .dev, and .post suffixes, add numbering
    return verstr



def get_git_revision_hash( short=True):
    def which(program):
    
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

        return None
    if not which('git'):
        print 'no git found on this system : can\'t get sha'
        return ""

    import subprocess
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    aubio_dir = os.path.join(this_file_dir, '..', '..')
    aubio_dir = os.path.abspath(aubio_dir)
    if not os.path.exists(aubio_dir):
        raise SystemError("git / root folder not found")
    gitcmd = ['git','-C',aubio_dir ,'rev-parse']
    if short:
      gitcmd.append('--short')
    gitcmd.append('HEAD')
    return subprocess.check_output(gitcmd).strip()

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
        print("checking for {:s}".format(package))
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

def add_local_aubio_sources(ext, usedouble = False):
    """ build aubio inside python module instead of linking against libaubio """
    print("Info: libaubio was not installed or built locally with waf, adding src/")
    aubio_sources = sorted(glob.glob(os.path.join('src', '**.c')))
    aubio_sources += sorted(glob.glob(os.path.join('src', '*', '**.c')))
    ext.sources += aubio_sources

def add_local_macros(ext, usedouble = False):
    # define macros (waf puts them in build/src/config.h)
    for define_macro in ['HAVE_STDLIB_H', 'HAVE_STDIO_H',
                         'HAVE_MATH_H', 'HAVE_STRING_H',
                         'HAVE_C99_VARARGS_MACROS',
                         'HAVE_LIMITS_H', 'HAVE_STDARG_H',
                         'HAVE_MEMCPY_HACKS']:
        ext.define_macros += [(define_macro, 1)]

def add_external_deps(ext, usedouble = False):
    # loof for additional packages
    print("Info: looking for *optional* additional packages")
    packages = ['libavcodec', 'libavformat', 'libavutil', 'libavresample',
                'jack',
                'jack',
                'sndfile',
                #'fftw3f',
               ]
    # samplerate only works with float
    if usedouble is False:
        packages += ['samplerate']
    else:
        print("Info: not adding libsamplerate in double precision mode")
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
    aubio_version = get_aubio_version()
    add_packages(['aubio = ' + aubio_version], ext)
    if 'aubio' not in ext.libraries:
        print("Info: aubio " + aubio_version + " was not found by pkg-config")
    else:
        print("Info: using system aubio " + aubio_version + " found in " + ' '.join(ext.library_dirs))

class CleanGenerated(distutils.command.clean.clean):
    def run(self):
        if os.path.isdir(output_path):
            distutils.dir_util.remove_tree(output_path)

from distutils.command.build_ext import build_ext as _build_ext
class build_ext(_build_ext):

    user_options = _build_ext.user_options + [
            # The format is (long option, short option, description).
            ('enable-double', None, 'use HAVE_AUBIO_DOUBLE=1 (default: 0)'),
            ]

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.enable_double = False

    def finalize_options(self):
        _build_ext.finalize_options(self)
        if self.enable_double:
            self.announce(
                    'will generate code for aubio compiled with HAVE_AUBIO_DOUBLE=1',
                    level=distutils.log.INFO)

    def build_extension(self, extension):
        if self.enable_double or 'HAVE_AUBIO_DOUBLE' in os.environ:
            extension.define_macros += [('HAVE_AUBIO_DOUBLE', 1)]
            enable_double = True
        else:
            enable_double = False
        # seack for aubio headers and lib in PKG_CONFIG_PATH
        add_system_aubio(extension)
        # the lib was not installed on this system
        if 'aubio' not in extension.libraries:
            # use local src/aubio.h
            if os.path.isfile(os.path.join('src', 'aubio.h')):
                add_local_aubio_header(extension)
            add_local_macros(extension)
            # look for a local waf build
            if os.path.isfile(os.path.join('build','src', 'fvec.c.1.o')):
                add_local_aubio_lib(extension)
            else:
                # check for external dependencies
                add_external_deps(extension, usedouble=enable_double)
                # add libaubio sources and look for optional deps with pkg-config
                add_local_aubio_sources(extension, usedouble=enable_double)
        # generate files python/gen/*.c, python/gen/aubio-generated.h
        extension.sources += generate_external(header, output_path, overwrite = False,
                usedouble=enable_double)
        return _build_ext.build_extension(self, extension)

#! /usr/bin/python
# 
# waf build script, see http://code.google.com/p/waf/
# usage:
#     $ waf distclean configure build
# get it:
#     $ svn co http://waf.googlecode.com/svn/trunk /path/to/waf
#     $ alias waf=/path/to/waf/waf-light
#
# TODO
#  - doc: add doxygen
#  - tests: move to new unit test system 

APPNAME = 'aubio'
VERSION = '0.3.3'
LIB_VERSION = '2.1.1'
srcdir = '.'
blddir = 'build'

def init(opt):
  pass

def set_options(opt):
  opt.add_option('--enable-double', action='store_true', default=False,
      help='compile aubio in double precision mode')
  opt.add_option('--disable-fftw3f', action='store_true', default=False,
      help='compile with fftw3 instead of fftw3f')
  opt.add_option('--enable-complex', action='store_true', default=False,
      help='compile with C99 complex')
  opt.add_option('--disable-jack', action='store_true', default=False,
      help='compile without jack support')
  opt.add_option('--disable-lash', action='store_true', default=False,
      help='compile without lash support')
  opt.add_option('--disable-libsamplerate', action='store_true', default=False,
      help='compile without libsamplerate support')
  opt.add_option('--with-target-platform', type='string',
      help='set target platform for cross-compilation', dest='target_platform')
  opt.tool_options('compiler_cc')
  opt.tool_options('compiler_cxx')
  opt.tool_options('gnu_dirs')
  opt.tool_options('UnitTest')

def configure(conf):
  import Options
  conf.check_tool('compiler_cc')
  conf.check_tool('compiler_cxx')
  conf.check_tool('gnu_dirs') # helpful for autotools transition and .pc generation
  conf.check_tool('misc') # needed for subst

  if Options.options.target_platform:
    Options.platform = Options.options.target_platform

  if Options.platform == 'win32':
    conf.env['shlib_PATTERN'] = 'lib%s.dll'

  # check for required headers
  conf.check(header_name='stdlib.h')
  conf.check(header_name='stdio.h')
  conf.check(header_name='math.h')
  conf.check(header_name='string.h')
  conf.check(header_name='limits.h')

  # optionally use complex.h
  if (Options.options.enable_complex == True):
    conf.check(header_name='complex.h')

  # check dependencies
  conf.check_cfg(package = 'sndfile', atleast_version = '1.0.4',
    args = '--cflags --libs')
  if (Options.options.disable_libsamplerate == False):
      conf.check_cfg(package = 'libsamplerate', atleast_version = '0.0.15',
        args = '--cflags --libs')

  # double precision mode
  if (Options.options.enable_double == True):
    conf.define('HAVE_AUBIO_DOUBLE', 1)
  else:
    conf.define('HAVE_AUBIO_DOUBLE', 0)

  # one of fftwf or fftw3f
  if (Options.options.disable_fftw3f == True):
    conf.check_cfg(package = 'fftw3', atleast_version = '3.0.0',
        args = '--cflags --libs')
  else:
    # fftw3f not disabled, take most sensible one according to enable_double
    if (Options.options.enable_double == True):
      conf.check_cfg(package = 'fftw3', atleast_version = '3.0.0',
          args = '--cflags --libs')
    else:
      conf.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
          args = '--cflags --libs')

  # optional dependancies
  if (Options.options.disable_jack == False):
    conf.check_cfg(package = 'jack', atleast_version = '0.15.0',
    args = '--cflags --libs')
  if (Options.options.disable_lash == False):
    conf.check_cfg(package = 'lash-1.0', atleast_version = '0.5.0',
    args = '--cflags --libs', uselib_store = 'LASH')

  # swig
  if conf.find_program('swig', var='SWIG', mandatory = False):
    conf.check_tool('swig', tooldir='swig')
    conf.check_swig_version('1.3.27')

    # python
    if conf.find_program('python', mandatory = False):
      conf.check_tool('python')
      conf.check_python_version((2,4,2))
      conf.check_python_headers()

  # check support for C99 __VA_ARGS__ macros
  check_c99_varargs = '''
#include <stdio.h>
#define AUBIO_ERR(...) fprintf(stderr, __VA_ARGS__)
'''
  if conf.check_cc(fragment = check_c99_varargs, 
      type='cstaticlib', 
      msg = 'Checking for C99 __VA_ARGS__ macro'):
    conf.define('HAVE_C99_VARARGS_MACROS', 1)

  # write configuration header
  conf.write_config_header('src/config.h')

  # add some defines used in examples 
  conf.define('AUBIO_PREFIX', conf.env['PREFIX'])
  conf.define('PACKAGE', APPNAME)

  # check if docbook-to-man is installed, optional
  conf.find_program('docbook-to-man', var='DOCBOOKTOMAN', mandatory=False)

def build(bld):
  bld.env['VERSION'] = VERSION 
  bld.env['LIB_VERSION'] = LIB_VERSION 

  build_extras(bld)

  # add sub directories
  bld.add_subdirs('src examples')
  if bld.env['SWIG']:
    if bld.env['PYTHON']:
      bld.add_subdirs('python/aubio python')

  # create the aubio.pc file for pkg-config
  aubiopc = bld.new_task_gen('subst')
  aubiopc.source = 'aubio.pc.in'
  aubiopc.target = 'aubio.pc'
  aubiopc.install_path = '${PREFIX}/lib/pkgconfig'

  # build manpages from sgml files
  if bld.env['DOCBOOKTOMAN']:
    import TaskGen
    TaskGen.declare_chain(
        name    = 'docbooktoman',
        rule    = '${DOCBOOKTOMAN} ${SRC} > ${TGT}',
        ext_in  = '.sgml',
        ext_out = '.1',
        reentrant = 0,
    )
    manpages = bld.new_task_gen(name = 'docbooktoman', 
        source=bld.path.ant_glob('doc/*.sgml'))
    bld.install_files('${MANDIR}/man1', bld.path.ant_glob('doc/*.1'))

  # install woodblock sound
  bld.install_files('${PREFIX}/share/sounds/aubio/', 
      'sounds/woodblock.aiff')

  # build and run the unit tests
  build_tests(bld)
  import UnitTest
  bld.add_post_fun(UnitTest.summary)

def shutdown(bld):
  pass

# loop over all *.c filenames in tests/src to build them all
# target name is filename.c without the .c
def build_tests(bld):
  for target_name in bld.path.ant_glob('tests/src/**/*.c').split():
    this_target = bld.new_task_gen(
        features = 'cc cprogram test',
        source = target_name,
        target = target_name.split('.')[0],
        includes = 'src',
        defines = 'AUBIO_UNSTABLE_API=1',
        uselib_local = 'aubio')
    # phasevoc-jack also needs jack 
    if target_name.endswith('test-phasevoc-jack.c'):
      this_target.includes = ['src', 'examples']
      this_target.uselib_local = ['aubio']
      this_target.uselib = ['JACK']
      this_target.source += ' examples/jackio.c'

def build_extras(bld):
    # corner cases to build these ones only once
    sndfileio = bld.new_task_gen(features = 'cc',
        includes = 'examples src',
        source = ['examples/sndfileio.c'], 
        target = 'sndfileio')

    defines = ['AUBIO_PREFIX="' + bld.env['AUBIO_PREFIX'] + '"']
    defines += ['PACKAGE="' + bld.env['PACKAGE'] + '"']

    utilsio = bld.new_task_gen(features = 'cc',
          includes = 'examples src',
          add_objects = 'sndfileio',
          source = ['examples/utils.c', 'examples/jackio.c'], 
          uselib = ['LASH', 'JACK', 'SNDFILE'],
          defines = defines, 
          target = 'utilsio')

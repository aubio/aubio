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

# read from VERSION
for l in open('VERSION').readlines(): exec (l.strip())

VERSION = '.'.join \
	([str(x) for x in [AUBIO_MAJOR_VERSION, AUBIO_MINOR_VERSION, AUBIO_PATCH_VERSION]]) \
	+ AUBIO_VERSION_STATUS
LIB_VERSION = '.'.join \
	([str(x) for x in [LIBAUBIO_LT_CUR, LIBAUBIO_LT_REV, LIBAUBIO_LT_AGE]])

import os.path, sys
if os.path.exists('src/config.h') or os.path.exists('Makefile'):
    print "Please run 'make distclean' to clean-up autotools files before using waf"
    sys.exit(1)

top = '.'
out = 'build'

def options(ctx):
  ctx.add_option('--enable-double', action='store_true', default=False,
      help='compile aubio in double precision mode')
  ctx.add_option('--enable-fftw', action='store_true', default=False,
      help='compile with ooura instead of fftw')
  ctx.add_option('--enable-fftw3f', action='store_true', default=False,
      help='compile with fftw3 instead of fftw3f')
  ctx.add_option('--enable-complex', action='store_true', default=False,
      help='compile with C99 complex')
  ctx.add_option('--enable-jack', action='store_true', default=False,
      help='compile with jack support')
  ctx.add_option('--enable-lash', action='store_true', default=False,
      help='compile with lash support')
  ctx.add_option('--enable-sndfile', action='store_true', default=False,
      help='compile with libsndfile support')
  ctx.add_option('--enable-samplerate', action='store_true', default=False,
      help='compile with libsamplerate support')
  ctx.add_option('--enable-swig', action='store_true', default=False,
      help='compile with swig support (obsolete)')
  ctx.add_option('--with-target-platform', type='string',
      help='set target platform for cross-compilation', dest='target_platform')
  ctx.load('compiler_c')
  ctx.load('compiler_cxx')
  ctx.load('gnu_dirs')
  ctx.load('waf_unit_test')

def configure(ctx):
  import Options
  ctx.check_tool('compiler_c')
  ctx.check_tool('compiler_cxx')
  ctx.check_tool('gnu_dirs') # helpful for autotools transition and .pc generation
  #ctx.check_tool('misc') # needed for subst
  ctx.load('waf_unit_test')
  ctx.env.CFLAGS = ['-g']

  if Options.options.target_platform:
    Options.platform = Options.options.target_platform

  if Options.platform == 'win32':
    ctx.env['shlib_PATTERN'] = 'lib%s.dll'

  # check for required headers
  ctx.check(header_name='stdlib.h')
  ctx.check(header_name='stdio.h')
  ctx.check(header_name='math.h')
  ctx.check(header_name='string.h')
  ctx.check(header_name='limits.h')

  # optionally use complex.h
  if (Options.options.enable_complex == True):
    ctx.check(header_name='complex.h')

  # check dependencies
  if (Options.options.enable_sndfile == True):
    ctx.check_cfg(package = 'sndfile', atleast_version = '1.0.4',
      args = '--cflags --libs')
  if (Options.options.enable_samplerate == True):
      ctx.check_cfg(package = 'samplerate', atleast_version = '0.0.15',
        args = '--cflags --libs')

  # double precision mode
  if (Options.options.enable_double == True):
    ctx.define('HAVE_AUBIO_DOUBLE', 1)
  else:
    ctx.define('HAVE_AUBIO_DOUBLE', 0)

  # check if pkg-config is installed, optional
  try:
    ctx.find_program('pkg-config', var='PKGCONFIG')
  except ctx.errors.ConfigurationError:
    ctx.msg('Could not find pkg-config', 'disabling fftw, jack, and lash')
    ctx.msg('Could not find fftw', 'using ooura')

  # optional dependancies using pkg-config
  if ctx.env['PKGCONFIG']:

    if (Options.options.enable_fftw == True or Options.options.enable_fftw3f == True):
      # one of fftwf or fftw3f
      if (Options.options.enable_fftw3f == True):
        ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
            args = '--cflags --libs')
        if (Options.options.enable_double == True):
          ctx.msg('Warning', 'fftw3f enabled, but aubio compiled in double precision!')
      else:
        # fftw3f not enabled, take most sensible one according to enable_double
        if (Options.options.enable_double == True):
          ctx.check_cfg(package = 'fftw3', atleast_version = '3.0.0',
              args = '--cflags --libs')
        else:
          ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
              args = '--cflags --libs')
      ctx.define('HAVE_FFTW3', 1)
    else:
      # fftw disabled, use ooura
      ctx.msg('Checking for FFT implementation', 'ooura')
      pass

    if (Options.options.enable_jack == True):
      ctx.check_cfg(package = 'jack', atleast_version = '0.15.0',
      args = '--cflags --libs')

    if (Options.options.enable_lash == True):
      ctx.check_cfg(package = 'lash-1.0', atleast_version = '0.5.0',
      args = '--cflags --libs', uselib_store = 'LASH')

  # swig
  if (Options.options.enable_swig == True):
    try:
      ctx.find_program('swig', var='SWIG')
    except ctx.errors.ConfigurationError:
      ctx.to_log('swig was not found, not looking for (ignoring)')

    if ctx.env['SWIG']:
      ctx.check_tool('swig')
      ctx.check_swig_version()

      # python
      if ctx.find_program('python'):
        ctx.check_tool('python')
        ctx.check_python_version((2,4,2))
        ctx.check_python_headers()

  # check support for C99 __VA_ARGS__ macros
  check_c99_varargs = '''
#include <stdio.h>
#define AUBIO_ERR(...) fprintf(stderr, __VA_ARGS__)
'''
  if ctx.check_cc(fragment = check_c99_varargs,
      type='cstlib',
      msg = 'Checking for C99 __VA_ARGS__ macro'):
    ctx.define('HAVE_C99_VARARGS_MACROS', 1)

  # write configuration header
  ctx.write_config_header('src/config.h')

  # add some defines used in examples
  ctx.define('AUBIO_PREFIX', ctx.env['PREFIX'])
  ctx.define('PACKAGE', APPNAME)

  # check if docbook-to-man is installed, optional
  try:
    ctx.find_program('docbook-to-man', var='DOCBOOKTOMAN')
  except ctx.errors.ConfigurationError:
    ctx.to_log('docbook-to-man was not found (ignoring)')

def build(ctx):
  ctx.env['VERSION'] = VERSION
  ctx.env['LIB_VERSION'] = LIB_VERSION

  # add sub directories
  ctx.add_subdirs(['src','examples'])
  if ctx.env['SWIG']:
    if ctx.env['PYTHON']:
      ctx.add_subdirs('python')

  # create the aubio.pc file for pkg-config
  if ctx.env['TARGET_PLATFORM'] == 'linux':
    aubiopc = ctx.new_task_gen('subst')
    aubiopc.source = 'aubio.pc.in'
    aubiopc.target = 'aubio.pc'
    aubiopc.install_path = '${PREFIX}/lib/pkgconfig'

  # build manpages from sgml files
  if ctx.env['DOCBOOKTOMAN']:
    import TaskGen
    TaskGen.declare_chain(
        name    = 'docbooktoman',
        rule    = '${DOCBOOKTOMAN} ${SRC} > ${TGT}',
        ext_in  = '.sgml',
        ext_out = '.1',
        reentrant = 0,
    )
    manpages = ctx.new_task_gen(name = 'docbooktoman',
        source=ctx.path.ant_glob('doc/*.sgml'))
    ctx.install_files('${MANDIR}/man1', ctx.path.ant_glob('doc/*.1'))

  # install woodblock sound
  ctx.install_files('${PREFIX}/share/sounds/aubio/',
      'sounds/woodblock.aiff')

  # build and run the unit tests
  build_tests(ctx)

def shutdown(ctx):
  pass

# loop over all *.c filenames in tests/src to build them all
# target name is filename.c without the .c
def build_tests(ctx):
  for target_name in ctx.path.ant_glob('tests/src/**/*.c'):
    uselib = []
    includes = ['src']
    extra_source = []
    if str(target_name).endswith('-jack.c') and ctx.env['JACK']:
      uselib += ['JACK']
      includes += ['examples']
      extra_source += ['examples/jackio.c']

    this_target = ctx.new_task_gen(
        features = 'c cprogram test',
        uselib = uselib,
        source = [target_name] + extra_source,
        target = str(target_name).split('.')[0],
        includes = includes,
        defines = 'AUBIO_UNSTABLE_API=1',
        cflags = ['-g'],
        use = 'aubio')

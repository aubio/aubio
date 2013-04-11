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
  ctx.add_option('--enable-fftw3f', action='store_true', default=False,
      help='compile with fftw3f instead of ooura (recommended)')
  ctx.add_option('--enable-fftw3', action='store_true', default=False,
      help='compile with fftw3 instead of ooura (recommended in double precision)')
  ctx.add_option('--enable-complex', action='store_true', default=False,
      help='compile with C99 complex')
  ctx.add_option('--enable-jack', action='store_true', default=None,
      help='compile with jack support')
  ctx.add_option('--enable-lash', action='store_true', default=None,
      help='compile with lash support')
  ctx.add_option('--enable-sndfile', action='store_true', default=None,
      help='compile with libsndfile support')
  ctx.add_option('--enable-samplerate', action='store_true', default=None,
      help='compile with libsamplerate support')
  ctx.add_option('--with-target-platform', type='string',
      help='set target platform for cross-compilation', dest='target_platform')
  ctx.load('compiler_c')
  ctx.load('waf_unit_test')

def configure(ctx):
  from waflib import Options
  ctx.load('compiler_c')
  ctx.load('waf_unit_test')
  ctx.env.CFLAGS += ['-g', '-Wall', '-Wextra']

  if Options.options.target_platform:
    Options.platform = Options.options.target_platform

  if Options.platform == 'win32':
    ctx.env['shlib_PATTERN'] = 'lib%s.dll'

  if Options.platform == 'darwin':
    ctx.env.CFLAGS += ['-arch', 'i386', '-arch', 'x86_64']
    ctx.env.LINKFLAGS += ['-arch', 'i386', '-arch', 'x86_64']
    ctx.env.CC = 'llvm-gcc-4.2'
    ctx.env.LINK_CC = 'llvm-gcc-4.2'
    ctx.env.FRAMEWORK = ['CoreFoundation', 'AudioToolbox', 'Accelerate']
    ctx.define('HAVE_ACCELERATE', 1)

  if Options.platform == 'ios':
    ctx.env.CC = 'clang'
    ctx.env.LD = 'clang'
    ctx.env.LINK_CC = 'clang'
    SDKVER="6.1"
    DEVROOT="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer"
    SDKROOT="%(DEVROOT)s/SDKs/iPhoneOS%(SDKVER)s.sdk" % locals()
    ctx.env.FRAMEWORK = ['CoreFoundation', 'AudioToolbox', 'Accelerate']
    ctx.define('HAVE_ACCELERATE', 1)
    ctx.env.CFLAGS += [ '-miphoneos-version-min=6.1', '-arch', 'armv7',
            '--sysroot=%s' % SDKROOT]
    ctx.env.LINKFLAGS += ['-std=c99', '-arch', 'armv7', '--sysroot=%s' %
            SDKROOT]

  # check for required headers
  ctx.check(header_name='stdlib.h')
  ctx.check(header_name='stdio.h')
  ctx.check(header_name='math.h')
  ctx.check(header_name='string.h')
  ctx.check(header_name='limits.h')

  # check support for C99 __VA_ARGS__ macros
  check_c99_varargs = '''
#include <stdio.h>
#define AUBIO_ERR(...) fprintf(stderr, __VA_ARGS__)
'''
  if ctx.check_cc(fragment = check_c99_varargs,
      type='cstlib',
      msg = 'Checking for C99 __VA_ARGS__ macro'):
    ctx.define('HAVE_C99_VARARGS_MACROS', 1)

  # optionally use complex.h
  if (Options.options.enable_complex == True):
    ctx.check(header_name='complex.h')

  # check dependencies
  if (Options.options.enable_sndfile != False):
      ctx.check_cfg(package = 'sndfile', atleast_version = '1.0.4',
        args = '--cflags --libs', mandatory = False)
  if (Options.options.enable_samplerate != False):
      ctx.check_cfg(package = 'samplerate', atleast_version = '0.0.15',
        args = '--cflags --libs', mandatory = False)

  # double precision mode
  if (Options.options.enable_double == True):
    ctx.define('HAVE_AUBIO_DOUBLE', 1)
  else:
    ctx.define('HAVE_AUBIO_DOUBLE', 0)

  # optional dependancies using pkg-config
  if (Options.options.enable_fftw3 != False or Options.options.enable_fftw3f != False):
    # one of fftwf or fftw3f
    if (Options.options.enable_fftw3f != False):
      ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
          args = '--cflags --libs', mandatory = False)
      if (Options.options.enable_double == True):
        ctx.msg('Warning', 'fftw3f enabled, but aubio compiled in double precision!')
    else:
      # fftw3f not enabled, take most sensible one according to enable_double
      if (Options.options.enable_double == True):
        ctx.check_cfg(package = 'fftw3', atleast_version = '3.0.0',
            args = '--cflags --libs', mandatory = False)
      else:
        ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
            args = '--cflags --libs', mandatory = False)
    ctx.define('HAVE_FFTW3', 1)
  else:
    # fftw disabled, use ooura
    if 'HAVE_ACCELERATE' in ctx.env.define_key:
        ctx.msg('Checking for FFT implementation', 'vDSP')
    else:
        ctx.msg('Checking for FFT implementation', 'ooura')
    pass

  if (Options.options.enable_jack != False):
    ctx.check_cfg(package = 'jack', atleast_version = '0.15.0',
    args = '--cflags --libs', mandatory = False)

  if (Options.options.enable_lash != False):
    ctx.check_cfg(package = 'lash-1.0', atleast_version = '0.5.0',
    args = '--cflags --libs', uselib_store = 'LASH', mandatory = False)

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

def build(bld):
  bld.env['VERSION'] = VERSION
  bld.env['LIB_VERSION'] = LIB_VERSION

  # add sub directories
  bld.recurse('src')
  from waflib import Options
  if Options.platform != 'ios':
      bld.recurse('examples')
      bld.recurse('tests')

  """
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
  bld.install_files('${PREFIX}/share/sounds/aubio/',
      'sounds/woodblock.aiff')
  """

def shutdown(bld):
    from waflib import Options, Logs
    if Options.platform == 'ios':
          msg ='aubio built for ios, contact the author for a commercial license'
          Logs.pprint('RED', msg)
          msg ='   Paul Brossier <piem@aubio.org>'
          Logs.pprint('RED', msg)

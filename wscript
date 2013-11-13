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

def add_option_enable_disable(ctx, name, default = None, help_str = None, help_disable_str = None):
  if help_str == None:
      help_str = 'enable ' + name + ' support'
  if help_disable_str == None:
      help_disable_str = 'do not ' + help_str
  ctx.add_option('--enable-' + name, action = 'store_true', default = default,
          dest = 'enable_' + name,
          help = help_str)
  ctx.add_option('--disable-' + name, action = 'store_false',
          #default = default,
          dest = 'enable_' + name,
          help = help_disable_str )

def options(ctx):
  add_option_enable_disable(ctx, 'double', default = False,
          help_str = 'compile aubio in double precision mode')
  add_option_enable_disable(ctx, 'fftw3f', default = False,
          help_str = 'compile with fftw3f instead of ooura (recommended)', help_disable_str = 'do not compile with fftw3f')
  add_option_enable_disable(ctx, 'fftw3', default = False,
          help_str = 'compile with fftw3 instead of ooura', help_disable_str = 'do not compile with fftw3')
  add_option_enable_disable(ctx, 'complex', default = False,
          help_str ='compile with C99 complex', help_disable_str = 'do not use C99 complex (default)' )
  add_option_enable_disable(ctx, 'jack', default = None,
          help_str = 'compile with jack (auto)', help_disable_str = 'disable jack support')
  add_option_enable_disable(ctx, 'lash', default = None,
          help_str = 'compile with LASH (auto)', help_disable_str = 'disable LASH' )
  add_option_enable_disable(ctx, 'sndfile', default = None,
          help_str = 'compile with sndfile (auto)', help_disable_str = 'disable sndfile')
  add_option_enable_disable(ctx, 'samplerate', default = None,
          help_str = 'compile with samplerate (auto)', help_disable_str = 'disable samplerate')

  ctx.add_option('--with-target-platform', type='string',
      help='set target platform for cross-compilation', dest='target_platform')
  ctx.load('compiler_c')
  ctx.load('waf_unit_test')
  ctx.load('gnu_dirs')

def configure(ctx):
  from waflib import Options
  ctx.load('compiler_c')
  ctx.load('waf_unit_test')
  ctx.load('gnu_dirs')
  ctx.env.CFLAGS += ['-g', '-Wall', '-Wextra']

  if Options.options.target_platform:
    Options.platform = Options.options.target_platform
    ctx.env['DEST_OS'] = Options.platform

  if Options.platform == 'win32':
    ctx.env['shlib_PATTERN'] = 'lib%s.dll'

  if Options.platform == 'darwin':
    ctx.env.CFLAGS += ['-arch', 'i386', '-arch', 'x86_64']
    ctx.env.LINKFLAGS += ['-arch', 'i386', '-arch', 'x86_64']
    ctx.env.FRAMEWORK = ['CoreFoundation', 'AudioToolbox', 'Accelerate']
    ctx.define('HAVE_ACCELERATE', 1)

  if Options.platform in [ 'ios', 'iosimulator' ]:
    ctx.define('HAVE_ACCELERATE', 1)
    ctx.define('TARGET_OS_IPHONE', 1)
    ctx.env.FRAMEWORK = ['CoreFoundation', 'AudioToolbox', 'Accelerate']
    SDKVER="7.0"
    MINSDKVER="6.1"
    ctx.env.CFLAGS += ['-std=c99']
    if Options.platform == 'ios':
        DEVROOT="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer"
        SDKROOT="%(DEVROOT)s/SDKs/iPhoneOS%(SDKVER)s.sdk" % locals()
        ctx.env.CFLAGS += [ '-arch', 'armv7' ]
        ctx.env.CFLAGS += [ '-arch', 'armv7s' ]
        ctx.env.LINKFLAGS += ['-arch', 'armv7']
        ctx.env.LINKFLAGS += ['-arch', 'armv7s']
        ctx.env.CFLAGS += [ '-miphoneos-version-min=' + MINSDKVER ]
        ctx.env.LINKFLAGS += [ '-miphoneos-version-min=' + MINSDKVER ]
    else:
        DEVROOT="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer"
        SDKROOT="%(DEVROOT)s/SDKs/iPhoneSimulator%(SDKVER)s.sdk" % locals()
        ctx.env.CFLAGS += [ '-arch', 'i386' ]
        ctx.env.LINKFLAGS += ['-arch', 'i386']
        ctx.env.CFLAGS += [ '-arch', 'x86_64' ]
        ctx.env.LINKFLAGS += ['-arch', 'x86_64']
        ctx.env.CFLAGS += [ '-mios-simulator-version-min=' + MINSDKVER ]
        ctx.env.LINKFLAGS += [ '-mios-simulator-version-min=' + MINSDKVER ]
    ctx.env.CFLAGS += [ '-isysroot' , SDKROOT]
    ctx.env.LINKFLAGS += [ '-isysroot' , SDKROOT]

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

  # fftw disabled, use ooura
  if 'HAVE_FFTW3F' in ctx.env.define_key:
    ctx.msg('Checking for FFT implementation', 'fftw3f')
  elif 'HAVE_FFTW3' in ctx.env.define_key:
    ctx.msg('Checking for FFT implementation', 'fftw3')
  elif 'HAVE_ACCELERATE' in ctx.env.define_key:
    ctx.msg('Checking for FFT implementation', 'vDSP')
  else:
    ctx.msg('Checking for FFT implementation', 'ooura')

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
  if bld.env['DEST_OS'] not in ['ios', 'iosimulator']:
      bld.recurse('examples')
      bld.recurse('tests')

  """
  # install woodblock sound
  bld.install_files('${PREFIX}/share/sounds/aubio/',
      'sounds/woodblock.aiff')
  """

  bld( source = 'aubio.pc.in' )

  # build manpages from sgml files
  if bld.env['DOCBOOKTOMAN']:
    from waflib import TaskGen
    if 'MANDIR' not in bld.env:
      bld.env['MANDIR'] = bld.env['PREFIX'] + '/share/man'
    TaskGen.declare_chain(
        name      = 'docbooktoman',
        rule      = '${DOCBOOKTOMAN} ${SRC} > ${TGT}',
        ext_in    = '.sgml',
        ext_out   = '.1',
        reentrant = False,
        install_path =  '${MANDIR}/man1',
    )
    bld( source = bld.path.ant_glob('doc/*.sgml') )

  """
  bld(rule = 'doxygen ${SRC}', source = 'web.cfg') #, target = 'doc/web/index.html')
  """


def shutdown(bld):
    from waflib import Options, Logs
    if Options.platform in ['ios', 'iosimulator']:
          msg ='aubio built for ios, contact the author for a commercial license'
          Logs.pprint('RED', msg)
          msg ='   Paul Brossier <piem@aubio.org>'
          Logs.pprint('RED', msg)


def dist(ctx):
    ctx.excl  = ' **/.waf-1* **/*~ **/*.pyc **/*.swp **/.lock-w* **/.git*'
    ctx.excl += ' **/build/*'
    ctx.excl += ' **/python/gen **/python/build **/python/dist'
    ctx.excl += ' **/**.zip **/**.tar.bz2'
    ctx.excl += ' **/doc/full/*'
    ctx.excl += ' **/python/*.db'
    ctx.excl += ' **/python.old/*'

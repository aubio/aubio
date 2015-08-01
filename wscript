#! /usr/bin/python
#
# usage:
#   $ python waf --help
#
# example:
#   $ ./waf distclean configure build
#
# Note: aubio uses the waf build system, which relies on Python. Provided you
# have Python installed, you do *not* need to install anything to build aubio.
# For more info about waf, see http://code.google.com/p/waf/ .

APPNAME = 'aubio'

# source VERSION
for l in open('VERSION').readlines(): exec (l.strip())

VERSION = '.'.join ([str(x) for x in [
    AUBIO_MAJOR_VERSION,
    AUBIO_MINOR_VERSION,
    AUBIO_PATCH_VERSION
    ]]) + AUBIO_VERSION_STATUS

LIB_VERSION = '.'.join ([str(x) for x in [
    LIBAUBIO_LT_CUR,
    LIBAUBIO_LT_REV,
    LIBAUBIO_LT_AGE]])

top = '.'
out = 'build'

def add_option_enable_disable(ctx, name, default = None,
        help_str = None, help_disable_str = None):
    if help_str == None:
        help_str = 'enable ' + name + ' support'
    if help_disable_str == None:
        help_disable_str = 'do not ' + help_str
    ctx.add_option('--enable-' + name, action = 'store_true',
            default = default,
            dest = 'enable_' + name.replace('-','_'),
            help = help_str)
    ctx.add_option('--disable-' + name, action = 'store_false',
            #default = default,
            dest = 'enable_' + name.replace('-','_'),
            help = help_disable_str )

def options(ctx):
    add_option_enable_disable(ctx, 'fftw3f', default = False,
            help_str = 'compile with fftw3f instead of ooura (recommended)',
            help_disable_str = 'do not compile with fftw3f')
    add_option_enable_disable(ctx, 'fftw3', default = False,
            help_str = 'compile with fftw3 instead of ooura',
            help_disable_str = 'do not compile with fftw3')
    add_option_enable_disable(ctx, 'complex', default = False,
            help_str ='compile with C99 complex',
            help_disable_str = 'do not use C99 complex (default)' )
    add_option_enable_disable(ctx, 'jack', default = None,
            help_str = 'compile with jack (auto)',
            help_disable_str = 'disable jack support')
    add_option_enable_disable(ctx, 'sndfile', default = None,
            help_str = 'compile with sndfile (auto)',
            help_disable_str = 'disable sndfile')
    add_option_enable_disable(ctx, 'avcodec', default = None,
            help_str = 'compile with libavcodec (auto)',
            help_disable_str = 'disable libavcodec')
    add_option_enable_disable(ctx, 'samplerate', default = None,
            help_str = 'compile with samplerate (auto)',
            help_disable_str = 'disable samplerate')
    add_option_enable_disable(ctx, 'memcpy', default = True,
            help_str = 'use memcpy hacks (default)',
            help_disable_str = 'do not use memcpy hacks')
    add_option_enable_disable(ctx, 'double', default = False,
            help_str = 'compile in double precision mode',
            help_disable_str = 'compile in single precision mode (default)')
    add_option_enable_disable(ctx, 'fat', default = False,
            help_str = 'build fat binaries (darwin only)',
            help_disable_str = 'do not build fat binaries (default)')

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

    target_platform = Options.platform
    if ctx.options.target_platform:
        target_platform = ctx.options.target_platform
    ctx.env['DEST_OS'] = target_platform

    if 'CL.exe' not in ctx.env.CC[0]:
        ctx.env.CFLAGS += ['-g', '-Wall', '-Wextra']
    else:
        ctx.env.CFLAGS += ['-Wall']

    if target_platform not in ['win32', 'win64']:
        ctx.env.CFLAGS += ['-fPIC']
    else:
        ctx.define('HAVE_WIN_HACKS', 1)
        ctx.env['cshlib_PATTERN'] = 'lib%s.dll'

    if target_platform == 'darwin' and ctx.options.enable_fat:
        ctx.env.CFLAGS += ['-arch', 'i386', '-arch', 'x86_64']
        ctx.env.LINKFLAGS += ['-arch', 'i386', '-arch', 'x86_64']

    if target_platform in [ 'darwin', 'ios', 'iosimulator']:
        ctx.env.FRAMEWORK = ['CoreFoundation', 'AudioToolbox', 'Accelerate']
        ctx.define('HAVE_SOURCE_APPLE_AUDIO', 1)
        ctx.define('HAVE_SINK_APPLE_AUDIO', 1)
        ctx.define('HAVE_ACCELERATE', 1)

    if target_platform in [ 'ios', 'iosimulator' ]:
        ctx.define('TARGET_OS_IPHONE', 1)
        MINSDKVER="6.1"
        ctx.env.CFLAGS += ['-std=c99']
        if target_platform == 'ios':
            DEVROOT = "/Applications/Xcode.app/Contents"
            DEVROOT += "/Developer/Platforms/iPhoneOS.platform/Developer"
            SDKROOT = "%(DEVROOT)s/SDKs/iPhoneOS.sdk" % locals()
            ctx.env.CFLAGS += [ '-arch', 'arm64' ]
            ctx.env.CFLAGS += [ '-arch', 'armv7' ]
            ctx.env.CFLAGS += [ '-arch', 'armv7s' ]
            ctx.env.LINKFLAGS += [ '-arch', 'arm64' ]
            ctx.env.LINKFLAGS += ['-arch', 'armv7']
            ctx.env.LINKFLAGS += ['-arch', 'armv7s']
            ctx.env.CFLAGS += [ '-miphoneos-version-min=' + MINSDKVER ]
            ctx.env.LINKFLAGS += [ '-miphoneos-version-min=' + MINSDKVER ]
        else:
            DEVROOT = "/Applications/Xcode.app/Contents"
            DEVROOT += "/Developer/Platforms/iPhoneSimulator.platform/Developer"
            SDKROOT = "%(DEVROOT)s/SDKs/iPhoneSimulator.sdk" % locals()
            ctx.env.CFLAGS += [ '-arch', 'i386' ]
            ctx.env.CFLAGS += [ '-arch', 'x86_64' ]
            ctx.env.LINKFLAGS += ['-arch', 'i386']
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
            msg = 'Checking for C99 __VA_ARGS__ macro',
            mandatory = False):
        ctx.define('HAVE_C99_VARARGS_MACROS', 1)

    # double precision mode
    if (ctx.options.enable_double == True):
        ctx.define('HAVE_AUBIO_DOUBLE', 1)
    else:
        ctx.define('HAVE_AUBIO_DOUBLE', 0)

    # optionally use complex.h
    if (ctx.options.enable_complex == True):
        ctx.check(header_name='complex.h')

    # check for fftw3
    if (ctx.options.enable_fftw3 != False or ctx.options.enable_fftw3f != False):
        # one of fftwf or fftw3f
        if (ctx.options.enable_fftw3f != False):
            ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
                    args = '--cflags --libs', mandatory = False)
            if (ctx.options.enable_double == True):
                ctx.msg('Warning', 'fftw3f enabled, but compiling in double precision!')
        else:
            # fftw3f not enabled, take most sensible one according to enable_double
            if (ctx.options.enable_double == True):
                ctx.check_cfg(package = 'fftw3', atleast_version = '3.0.0',
                        args = '--cflags --libs', mandatory = False)
            else:
                ctx.check_cfg(package = 'fftw3f', atleast_version = '3.0.0',
                        args = '--cflags --libs', mandatory = False)
        ctx.define('HAVE_FFTW3', 1)

    # fftw not enabled, use vDSP or ooura
    if 'HAVE_FFTW3F' in ctx.env.define_key:
        ctx.msg('Checking for FFT implementation', 'fftw3f')
    elif 'HAVE_FFTW3' in ctx.env.define_key:
        ctx.msg('Checking for FFT implementation', 'fftw3')
    elif 'HAVE_ACCELERATE' in ctx.env.define_key:
        ctx.msg('Checking for FFT implementation', 'vDSP')
    else:
        ctx.msg('Checking for FFT implementation', 'ooura')

    # check for libsndfile
    if (ctx.options.enable_sndfile != False):
        ctx.check_cfg(package = 'sndfile', atleast_version = '1.0.4',
                args = '--cflags --libs', mandatory = False)

    # check for libsamplerate
    if (ctx.options.enable_samplerate != False):
        ctx.check_cfg(package = 'samplerate', atleast_version = '0.0.15',
                args = '--cflags --libs', mandatory = False)

    # check for jack
    if (ctx.options.enable_jack != False):
        ctx.check_cfg(package = 'jack',
                args = '--cflags --libs', mandatory = False)

    # check for libav
    if (ctx.options.enable_avcodec != False):
        ctx.check_cfg(package = 'libavcodec', atleast_version = '54.35.0',
                args = '--cflags --libs', uselib_store = 'AVCODEC', mandatory = False)
        ctx.check_cfg(package = 'libavformat', atleast_version = '52.3.0',
                args = '--cflags --libs', uselib_store = 'AVFORMAT', mandatory = False)
        ctx.check_cfg(package = 'libavutil', atleast_version = '52.3.0',
                args = '--cflags --libs', uselib_store = 'AVUTIL', mandatory = False)
        ctx.check_cfg(package = 'libavresample', atleast_version = '1.0.1',
                args = '--cflags --libs', uselib_store = 'AVRESAMPLE', mandatory = False)
        if all ( 'HAVE_' + i in ctx.env.define_key
                for i in ['AVCODEC', 'AVFORMAT', 'AVUTIL', 'AVRESAMPLE'] ):
            ctx.define('HAVE_LIBAV', 1)
            ctx.msg('Checking for all libav libraries', 'yes')
        else:
            ctx.msg('Checking for all libav libraries', 'not found', color = 'YELLOW')

    ctx.define('HAVE_WAVREAD', 1)
    ctx.define('HAVE_WAVWRITE', 1)

    # use memcpy hacks
    if (ctx.options.enable_memcpy == True):
        ctx.define('HAVE_MEMCPY_HACKS', 1)
    else:
        ctx.define('HAVE_MEMCPY_HACKS', 0)

    # write configuration header
    ctx.write_config_header('src/config.h')

    # add some defines used in examples
    ctx.define('AUBIO_PREFIX', ctx.env['PREFIX'])
    ctx.define('PACKAGE', APPNAME)

    # check if txt2man is installed, optional
    try:
      ctx.find_program('txt2man', var='TXT2MAN')
    except ctx.errors.ConfigurationError:
      ctx.to_log('txt2man was not found (ignoring)')

    # check if doxygen is installed, optional
    try:
      ctx.find_program('doxygen', var='DOXYGEN')
    except ctx.errors.ConfigurationError:
      ctx.to_log('doxygen was not found (ignoring)')

def build(bld):
    bld.env['VERSION'] = VERSION
    bld.env['LIB_VERSION'] = LIB_VERSION

    # add sub directories
    bld.recurse('src')
    if bld.env['DEST_OS'] not in ['ios', 'iosimulator']:
        pass
    if bld.env['DEST_OS'] not in ['ios', 'iosimulator', 'android']:
        bld.recurse('examples')
        bld.recurse('tests')

    bld( source = 'aubio.pc.in' )

    # build manpages from txt files using txt2man
    if bld.env['TXT2MAN']:
        from waflib import TaskGen
        if 'MANDIR' not in bld.env:
            bld.env['MANDIR'] = bld.env['PREFIX'] + '/share/man'
        rule_str = '${TXT2MAN} -t `basename ${TGT} | cut -f 1 -d . | tr a-z A-Z`'
        rule_str += ' -r ${PACKAGE}\\ ${VERSION} -P ${PACKAGE}'
        rule_str += ' -v ${PACKAGE}\\ User\\\'s\\ manual'
        rule_str += ' -s 1 ${SRC} > ${TGT}'
        TaskGen.declare_chain(
                name      = 'txt2man',
                rule      = rule_str,
                ext_in    = '.txt',
                ext_out   = '.1',
                reentrant = False,
                install_path =  '${MANDIR}/man1',
                )
        bld( source = bld.path.ant_glob('doc/*.txt') )

    # build documentation from source files using doxygen
    if bld.env['DOXYGEN']:
        bld( name = 'doxygen', rule = 'doxygen ${SRC} > /dev/null',
                source = 'doc/web.cfg',
                cwd = 'doc')
        bld.install_files( '${PREFIX}' + '/share/doc/libaubio-doc',
                bld.path.ant_glob('doc/web/html/**'),
                cwd = bld.path.find_dir ('doc/web'),
                relative_trick = True)

def shutdown(bld):
    from waflib import Logs
    if bld.options.target_platform in ['ios', 'iosimulator']:
        msg ='building for %s, contact the author for a commercial license' % bld.options.target_platform
        Logs.pprint('RED', msg)
        msg ='   Paul Brossier <piem@aubio.org>'
        Logs.pprint('RED', msg)

def dist(ctx):
    ctx.excl  = ' **/.waf-1* **/*~ **/*.pyc **/*.swp **/.lock-w* **/.git*'
    ctx.excl += ' **/build/*'
    ctx.excl += ' **/python/gen **/python/build **/python/dist'
    ctx.excl += ' **/**.zip **/**.tar.bz2'
    ctx.excl += ' **/doc/full/* **/doc/web/*'
    ctx.excl += ' **/python/*.db'
    ctx.excl += ' **/python.old/*'
    ctx.excl += ' **/python/tests/sounds'

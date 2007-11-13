dnl Process this file with autoconf to produce a configure script

dnl Init autoconf
AC_INIT(src/aubio.h)

dnl Package name and version
PACKAGE=aubio
. $srcdir/VERSION
AUBIO_VERSION=${AUBIO_MAJOR_VERSION}.${AUBIO_MINOR_VERSION}.${AUBIO_PATCH_VERSION}${AUBIO_VERSION_STATUS}
VERSION=${AUBIO_VERSION}

dnl Shared library version
SHARED_VERSION_INFO=${LIBAUBIO_LT_CUR}:${LIBAUBIO_LT_REV}:${LIBAUBIO_LT_AGE}
AC_SUBST(SHARED_VERSION_INFO)

dnl Init automake
AM_INIT_AUTOMAKE(${PACKAGE}, ${VERSION})

dnl Maintainer mode
AM_MAINTAINER_MODE

dnl Guess the host
AC_CANONICAL_HOST

dnl Check for programs
AC_PROG_CC
AM_PROG_CC_C_O dnl compiling with per-target flag 
if test "$ac_cv_prog_cc" = "no" ; then
   AC_MSG_ERROR([*** No C compiler found !])
fi
AC_PROG_INSTALL

AUBIO_CFLAGS=

dnl Enable debugging (no)
AC_ARG_ENABLE(debug,
  AC_HELP_STRING([--enable-debug],[compile in debug mode [[default=no]]]),
  with_debug="yes",
  with_debug="no")
if test "$with_debug" = "yes"
then
  AC_DEFINE(DEBUG,1,[Define to enable debug])
  AUBIO_CFLAGS="$AUBIO_CFLAGS -g"
fi

dnl Enable full warnings (yes)
AC_ARG_ENABLE(warnings,
  AC_HELP_STRING([--enable-warnings],[compile with all gcc warnings [[default=yes]]]),
  with_warnme="no",
  with_warnme="yes")
if test "$with_warnme" = "yes"
then
  dnl Check for -Wextra support to allow compilation on gcc <= 3.4
  AC_CACHE_CHECK([for -Wextra option to $CC], ac_cv_cc_wextra, 
  [old_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -Wextra"
  AC_COMPILE_IFELSE([void foo (void) {}],
          ac_cv_cc_wextra=yes, ac_cv_cc_wextra=no)
  CFLAGS="$old_CFLAGS"
  ])
  if test "$ac_cv_cc_wextra" = "yes"
  then
    AUBIO_CFLAGS="$AUBIO_CFLAGS -Wall -Wextra"
  else 
    AUBIO_CFLAGS="$AUBIO_CFLAGS -Wall"
  fi
fi

dnl fail on compilation warnings 
AC_ARG_ENABLE(errorfail,
  AC_HELP_STRING([--enable-errorfail],[fail on compilation warnings [[default=no]]]),
  AUBIO_CFLAGS="$AUBIO_CFLAGS -Werror -Wmissing-prototypes -Wmissing-declarations -Wno-unused-parameter",
  with_errorfail="no")

dnl add gcov/lcov profiling and coverage flags
AC_ARG_ENABLE(lcov,
  AC_HELP_STRING([--enable-lcov],[compile with gcov/lcov profiling flags [[default=no]]]),
  AUBIO_CFLAGS="$AUBIO_CFLAGS -fprofile-arcs -ftest-coverage",
  with_lcov="no")

dnl Check for libtool
AC_LIBTOOL_DLOPEN
dnl AC_DISABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS(src/config.h)
AC_CONFIG_FILES(aubio.pc)

AM_CONDITIONAL(MINGW, false)
AM_CONDITIONAL(DARWIN, false)
case "${build_os}" in
*mingw* | *cygwin*)
  mingw32_support="yes"
  AC_CHECK_HEADER(windows.h)
  AM_CONDITIONAL(MINGW, true)
  LDFLAGS="$LDFLAGS -no-undefined"
  ;;
*darwin* | *rhapsody* | *macosx*)
  dnl on macosx, cosf went to -lmx
  LDFLAGS="$LDFLAGS -lmx"
  dnl as long double doesn't sit well with -Wall -Werror
  AUBIO_CFLAGS="$AUBIO_CFLAGS -Wno-long-double"
  dnl Workaround darwin autoconf bug
  CPPFLAGS="$CPPFLAGS -I${prefix}/include"
  AC_ISC_POSIX
  AM_CONDITIONAL(DARWIN, true)
  ;;
*)
  AC_ISC_POSIX
  ;;
esac

dnl Check for required libraries
AC_CHECK_LIB(pthread, pthread_create)

dnl Check for header files
AC_HEADER_STDC
AC_CHECK_HEADERS([string.h stdlib.h stdio.h math.h errno.h stdarg.h unistd.h signal.h],,)
AC_CHECK_HEADERS(fftw3.h,,AC_MSG_ERROR([Ouch! missing fftw3.h header]))
AC_ARG_ENABLE(complex,
  AC_HELP_STRING([--enable-complex],[compile with complex.h [[default=auto]]]),
  [with_complex=$enableval],
  with_complex="yes")
if test "$with_complex" = "yes"; then
  AC_CHECK_HEADERS(complex.h,,AC_MSG_WARN([Ouch! missing complex.h header]))
fi

dnl Check for __VAR_ARGS__ support
AC_CACHE_CHECK(for C99 __VA_ARGS__ macro,
    ac_cv_varargs_macros,
AC_TRY_COMPILE([
  #include <stdio.h>
  #define AUBIO_ERR(...)                       fprintf(stderr, __VA_ARGS__)
],
[
  AUBIO_ERR("%s\n", "ERR");
],
        ac_cv_varargs_macros=yes,
        ac_cv_varargs_macros=no)
)
if test "$ac_cv_varargs_macros" = "yes"; then
    AC_DEFINE(HAVE_C99_VARARGS_MACROS, 1,
            [Defined when c99 style varargs macros are supported])
fi

dnl Check for pkg-config
AC_PATH_PROG(PKG_CONFIG,pkg-config,no)

PKG_CHECK_MODULES(SNDLIB,     sndfile >= 1.0.4,       SNDLIB_SUPPORT=1)
PKG_CHECK_MODULES(SAMPLERATE, samplerate  >= 0.0.15,  SAMPLERATE_SUPPORT=1)

if test "${SNDLIB_SUPPORT}" = "1"; then
  AC_DEFINE(SNDLIB_SUPPORT,1,[Define to enable libsndfile support])
fi

dnl Check for fftw3 (required)
AC_ARG_ENABLE(fftw3f,
  AC_HELP_STRING([--enable-fftw3f],[compile with fftw3f [[default=auto]]]),
  [with_fftw3f=$enableval],
  with_fftw3f="yes")
if test "$with_fftw3f" = "yes"; then
  PKG_CHECK_MODULES(FFTWLIB,    fftw3f >= 3.0.0,     FFTW3F_SUPPORT=1, FFTW3F_SUPPORT=0)
else
  PKG_CHECK_MODULES(FFTWLIB,    fftw3  >= 3.0.0,     FFTW3_SUPPORT=1)
fi
if test "${FFTW3F_SUPPORT}" = "0"; then
  PKG_CHECK_MODULES(FFTWLIB,    fftw3  >= 3.0.0,     FFTW3_SUPPORT=1)
fi
if test "${FFTW3_SUPPORT}" = "1"; then
  AC_DEFINE(FFTW3_SUPPORT,1,[Define to enable fftw3 support])
fi
if test "${FFTW3F_SUPPORT}" = "1"; then
  AC_DEFINE(FFTW3F_SUPPORT,1,[Define to enable fftw3f support])
fi

dnl Enable jack support (auto)
AC_ARG_ENABLE(jack,
  AC_HELP_STRING([--enable-jack],[compile with jack [[default=auto]]]),
  [with_jack=$enableval],
  with_jack="yes")
if test "$with_jack" = "yes"
then
  PKG_CHECK_MODULES(JACK,       jack  >= 0.15.0,     JACK_SUPPORT=1, JACK_SUPPORT=0)
  if test "${JACK_SUPPORT}" = "1"; then
    AC_DEFINE(JACK_SUPPORT,1,[Define to enable jack support])
  fi
fi

dnl Enable alsa support (auto)
AC_ARG_ENABLE(alsa,
  AC_HELP_STRING([--enable-alsa],[compile with alsa [[default=auto]]]),
  [with_alsa=$enableval],
  with_alsa="yes")
if test "$with_alsa" = "yes"
then
  if test "$with_jack" = "yes"
  then
    PKG_CHECK_MODULES(ALSA,      alsa >= 0.0.9,      ALSA_SUPPORT=1, ALSA_SUPPORT=0)
    if test "${ALSA_SUPPORT}" = "1"; then
      AC_DEFINE(ALSA_SUPPORT,1,[Define to enable alsa support])
    fi
  else
    AC_MSG_WARN([Disabling alsa as jack was not found])
  fi
fi

dnl Enable lash support
AC_ARG_ENABLE(lash,
  AC_HELP_STRING([--enable-lash],[compile with lash [[default=auto]]]),
  [with_lash=$enableval],
  with_lash="yes")
if test "$with_lash" = "yes"
then
  PKG_CHECK_MODULES(LASH,   lash-1.0 >= 0.5.0,   LASH_SUPPORT=1, LASH_SUPPORT=0)
  if test "${LASH_SUPPORT}" = "1"; then
    LASH_CFLAGS+="-DLASH_SUPPORT"
  fi
fi
  
dnl Enable unit tests 
AC_ARG_ENABLE(testprogs,
  AC_HELP_STRING([--enable-testprogs],[compile test programs [[default=no]]]),
  [with_testprogs=$enableval],
  with_testprogs="no")
AM_CONDITIONAL(COMPILE_TESTS,test "${with_testprogs}" != "no")

AC_SUBST(AUBIO_CFLAGS)

dnl Check for swig and python
dnl should check for swig version and python headers
AC_PATH_PROG(SWIG,swig,no)
AM_CONDITIONAL(SWIGFOUND, test "${SWIG}" != "no")
AM_PATH_PYTHON
AM_CONDITIONAL(PYTHONFOUND, test "${PYTHON}" != "no")

dnl Check for docbook-to-man
AC_PATH_PROG(DOCBOOK_TO_MAN,docbook-to-man,no)
AM_CONDITIONAL(DOCBOOKFOUND, test "${DOCBOOK_TO_MAN}" != "no")

dnl Check for Puredata
AC_CHECK_HEADER(m_pd.h,PUREDATA=y,AC_MSG_WARN([Puredata header not found.]))
AM_CONDITIONAL(PUREDATAFOUND, test "${PUREDATA}" = "y")

dnl Create Makefiles
AC_OUTPUT([
    Makefile
    src/Makefile
    ext/Makefile
    examples/Makefile
    tests/Makefile
    tests/src/Makefile
    sounds/Makefile
    swig/Makefile
    python/Makefile
    python/aubio/Makefile
    plugins/Makefile
    plugins/audacity/Makefile
    plugins/audacity/plug-ins/Makefile
    plugins/wavesurfer/Makefile
    plugins/puredata/Makefile
    doc/Makefile
  ])

dnl Print summary
echo
echo "**************************************************************"
echo "Summary:"
if test "${FFTW3F_SUPPORT}" = "1"; then
  echo "Fftw3:                   yes (using fftw3f)"
else
if test "${FFTW3_SUPPORT}" = "1"; then
  echo "Fftw3:                   yes (not using fftw3f)"
else
  echo "Fftw3:                   no (that should not happen)"
fi
fi
if test "${SNDLIB_SUPPORT}" = "1"; then
  echo "Libsndfile:              yes"
else
  echo "Libsndfile:              no"
fi
if test "${SAMPLERATE_SUPPORT}" = "1"; then
  echo "Libsamplerate:           yes"
else
  echo "Libsamplerate:           no"
fi
if test "${JACK_SUPPORT}" = "1"; then
  echo "JACK:                    yes"
else
  echo "JACK:                    no"
fi
if test "${ALSA_SUPPORT}" = "1"; then
  echo "ALSA midi:               yes"
else
  echo "ALSA midi:               no"
fi
if test "${LASH_SUPPORT}" = "1"; then
  echo "Lash:                    yes"
else
  echo "Lash:                    no"
fi
if test "${PUREDATA}" = "y"; then
  echo "PureData:                yes"
else
  echo "PureData:                no"
fi
echo "**************************************************************"
echo Configuration completed successfully. Type \'make\' to build ${PACKAGE}

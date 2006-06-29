dnl Process this file with autoconf to produce a configure script

dnl Init autoconf
AC_INIT(src/aubio.h)

dnl Package name and version
PACKAGE=aubio
. $srcdir/VERSION
AUBIO_VERSION=${AUBIO_MAJOR_VERSION}.${AUBIO_MINOR_VERSION}.${AUBIO_PATCH_VERSION}${AUBIO_VERSION_STATUS}
VERSION=${AUBIO_VERSION}

dnl Shared library version
LIBAUBIO_LT_CURRENT=2
LIBAUBIO_LT_REVISION=0
LIBAUBIO_LT_AGE=0
SHARED_VERSION_INFO="$LIBAUBIO_LT_CURRENT:$LIBAUBIO_LT_REVISION:$LIBAUBIO_LT_AGE"
AC_SUBST(SHARED_VERSION_INFO)

dnl Init automake
AM_INIT_AUTOMAKE(${PACKAGE}, ${VERSION})

dnl Maintainer mode
AM_MAINTAINER_MODE

dnl Guess the host
AC_CANONICAL_HOST

dnl Check for programs
AC_PROG_CC
if test "$ac_cv_prog_cc" = "no" ; then
   AC_MSG_ERROR([*** No C compiler found !])
fi
AC_PROG_INSTALL

dnl Default optimization
CFLAGS="$CFLAGS -O2"

dnl Enable debugging (no)
AC_ARG_ENABLE(debug,
  [  --enable-debug[[=value]]  compile with debug [[default=no]]],
  with_debug="yes",
  with_debug="no")
if test "$with_debug" = "yes"
then
  AC_DEFINE(DEBUG,1,[Define to enable debug])
  CFLAGS="$CFLAGS -g"
fi

dnl Enable full warnings (yes)
AC_ARG_ENABLE(warnme,
  [  --enable-warnme[[=value]] compile with all -W [[default=yes]]],
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
  ])
  if test "$ac_cv_cc_wextra" = "yes"
  then
    CFLAGS="$old_CFLAGS -Wall -Wextra"
  else 
    CFLAGS="$old_CFLAGS -Wall"
  fi
fi

AC_ARG_ENABLE(failonerror,
  [  --enable-fwarn[[=value]]  fail on compilation warnings [[default=yes]]],
  with_warnme="no",
  CFLAGS="$CFLAGS -Werror -Wmissing-prototypes -Wmissing-declarations -Wno-unused-parameter")

dnl Check for libtool
AC_LIBTOOL_DLOPEN
dnl AC_DISABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS(src/config.h)
AC_CONFIG_FILES(aubio.pc)

case "${host_os}" in
*mingw*)
  mingw32_support="yes"
  AC_CHECK_HEADER(windows.h)
  ;;
*darwin*)
  dnl on macosx, cosf went to -lmx
  LDFLAGS="$LDFLAGS -lmx"
  dnl as long double doesn't sit well with -Wall -Werror
  CFLAGS="$CFLAGS -Wno-long-double"
  AC_ISC_POSIX
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
AC_CHECK_HEADERS(complex.h,,AC_MSG_ERROR([Ouch! missing complex.h header]))
AC_CHECK_HEADERS(fftw3.h  ,,AC_MSG_ERROR([Ouch! missing fftw3.h header]))

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

dnl Enable ladcca support (obsolete)
AC_ARG_ENABLE(ladcca,
  AC_HELP_STRING([--enable-ladcca],[compile with ladcca [[default=no]]]),
  [with_ladcca=$enableval],
  with_ladcca="no")
if test "$with_ladcca" = "yes"
then
  PKG_CHECK_MODULES(LADCCA,   ladcca-1.0 >= 0.4.0,   LADCCA_SUPPORT=1)
  if test "${LADCCA_SUPPORT}" = "1"; then
    AC_DEFINE(LADCCA_SUPPORT,1,[Define to enable ladcca support])
  fi
fi
  
dnl Enable unit tests 
AC_ARG_ENABLE(testprogs,
  AC_HELP_STRING([--enable-testprogs],[compile test programs [[default=no]]]),
  [with_testprogs=$enableval],
  with_testprogs="no")
AM_CONDITIONAL(COMPILE_TESTS,test "${with_testprogs}" != "no")

dnl Check for swig and python
dnl should check for swig version and python headers
AC_PATH_PROG(SWIG,swig,no)
AM_CONDITIONAL(SWIGFOUND, test "${SWIG}" != "no")
AM_PATH_PYTHON
AM_CONDITIONAL(PYTHONFOUND, test "${PYTHON}" != "no")
#
#if PYTHONFOUND
case "${host_os}" in
*darwin*)
  SWCFLAGS="-Wno-long-double"
  SWLDFLAGS="-bundle -framework python -L${prefix}/lib/python${PYTHON_VERSION}/config -lpython${PYTHON_VERSION}"
  ;;
*)
  SWLDFLAGS="-shared"
  ;;
esac
#
AC_SUBST(SWCFLAGS)
AC_SUBST(SWLDFLAGS)

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
    examples/tests/Makefile
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
if test "${LADCCA_SUPPORT}" = "1"; then
  echo "Ladcca:                  yes"
fi
echo "**************************************************************"
echo Configuration completed successfully. Type \'make\' to build ${PACKAGE}

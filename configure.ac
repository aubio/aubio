dnl Process this file with autoconf to porduce a configure script

AC_INIT(src/aubio.h)

. $srcdir/VERSION
AUBIO_VERSION=${AUBIO_MAJOR_VERSION}.${AUBIO_MINOR_VERSION}.${AUBIO_PATCH_VERSION}${AUBIO_VERSION_STATUS}
VERSION=${AUBIO_VERSION}
PACKAGE=aubio

AM_INIT_AUTOMAKE(${PACKAGE}, ${VERSION})

AC_PREFIX_DEFAULT(/usr)

dnl enable optimization
CFLAGS="$CFLAGS -O2"

	
dnl Check for debug
AC_ARG_ENABLE(debug,
  [  --enable-debug[[=value]]   compile with debug [[default=no]]],
  with_debug="yes",
  with_debug="no")
if test "$with_debug" = "yes"
then
  AC_DEFINE(DEBUG,1,[Define to enable debug])
  CFLAGS="$CFLAGS -g"
fi

dnl
AC_ARG_ENABLE(warnme,
  [  --enable-warnme[[=value]]   compile with all -W [[default=yes]]],
  with_warnme="no",
  with_warnme="yes")
if test "$with_warnme" = "yes"
then
	ALL_CWARNINGS="-Wall -Werror -Wmissing-prototypes -Wmissing-declarations -Wno-char-subscripts -Wno-unknown-pragmas"
	CFLAGS="$CFLAGS $ALL_CWARNINGS"
fi


dnl Guess the host
AC_CANONICAL_HOST

dnl Check for programs
AC_PROG_CC
if test "$ac_cv_prog_cc" = "no" ; then
   AC_MSG_ERROR([*** No C compiler found !])
fi
AC_PROG_INSTALL
dnl AC_PROG_SWIG

dnl Check for libtool
AC_LIBTOOL_DLOPEN
dnl AC_DISABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS(src/config.h)
AC_CONFIG_FILES(aubio.pc)

case "${host_os}" in
mingw*)
  mingw32_support="yes"
  AC_CHECK_HEADER(windows.h)
  ;;
darwin*)
  dnl on macosx, cosf went to -lmx
  LDFLAGS="$LDFLAGS -lmx"
  AC_ISC_POSIX
  ;;
*)
  AC_ISC_POSIX
  ;;
esac

dnl Check for libraries
AC_CHECK_LIB(pthread, pthread_create)

dnl Check for header files
AC_HEADER_STDC
AC_CHECK_HEADERS([string.h stdlib.h stdio.h math.h errno.h stdarg.h unistd.h signal.h],,)
AC_CHECK_HEADERS(complex.h,,AC_MSG_ERROR([Ouch! missing complex.h header]))
AC_CHECK_HEADERS(fftw3.h  ,,AC_MSG_ERROR([Ouch! missing fftw3.h headeri]))

dnl check for pkg-config
AC_PATH_PROG(PKG_CONFIG,pkg-config,no)

dnl Check for required libraries
PKG_CHECK_MODULES(FFTWLIB,    fftw3  >= 3.0.0,	      FFTW3_SUPPORT=1)
if test "${FFTW3_SUPPORT}" = "1"; then
  AC_DEFINE(FFTW3_SUPPORT,1,[Define to enable fftw3 support])
fi
PKG_CHECK_MODULES(SNDLIB,     sndfile >= 1.0.4,	      SNDLIB_SUPPORT=1)
PKG_CHECK_MODULES(SAMPLERATE, samplerate  >= 0.0.15,  SAMPLERATE_SUPPORT=1)

if test "${SNDLIB_SUPPORT}" = "1"; then
  AC_DEFINE(SNDLIB_SUPPORT,1,[Define to enable libsndfile support])
fi


dnl Check for optional libraries
AC_ARG_ENABLE(fftw3f,
  [  --enable-fftw3f[[=value]]   compile with fftw3f [[default=no]]],
  with_fftw3f="yes",
  with_fftw3f="no")
if test "$with_fftw3f" = "yes"; then
	PKG_CHECK_MODULES(FFTWLIB,    fftw3f  >= 3.0.0,     FFTW3F_SUPPORT=1)
fi
if test "${FFTW3F_SUPPORT}" = "1"; then
  AC_DEFINE(FFTW3F_SUPPORT,1,[Define to enable fftw3f support])
fi

AC_ARG_ENABLE(jack,
  [  --enable-jack[[=value]]   compile with jack [[default=no]]],
  with_jack="yes",
  with_jack="no")
if test "$with_jack" = "yes"
then
	PKG_CHECK_MODULES(JACK,       jack  >= 0.15.0,	      JACK_SUPPORT=1)
	if test "${JACK_SUPPORT}" = "1"; then
	  AC_DEFINE(JACK_SUPPORT,1,[Define to enable jack support])
	fi
fi

AC_ARG_ENABLE(alsa,
  [  --enable-alsa[[=value]]   compile with alsa [[default=no]]],
  with_alsa="yes",
  with_alsa="no")
if test "$with_alsa" = "yes"
then
	PKG_CHECK_MODULES(ALSA,       alsa >= 0.0.9,          ALSA_SUPPORT=1)
	if test "${ALSA_SUPPORT}" = "1"; then
	  AC_DEFINE(ALSA_SUPPORT,1,[Define to enable alsa support])
	fi
fi

AC_ARG_ENABLE(ladcca,
  [  --enable-ladcca[[=value]]   compile with ladcca [[default=no]]],
  with_ladcca="yes",
  with_ladcca="no")
if test "$with_ladcca" = "yes"
then
	PKG_CHECK_MODULES(LADCCA,       ladcca-1.0 >= 0.4.0,          LADCCA_SUPPORT=1)
	if test "${LADCCA_SUPPORT}" = "1"; then
	  AC_DEFINE(LADCCA_SUPPORT,1,[Define to enable ladcca support])
	fi
fi
  
AM_PATH_PYTHON(,,)

AC_OUTPUT([
		Makefile
		src/Makefile
		examples/Makefile
		sounds/Makefile
		doc/Makefile
		python/Makefile
		python/aubio/Makefile
		plugins/Makefile
		plugins/audacity/Makefile
		plugins/audacity/plug-ins/Makefile
		plugins/wavesurfer/Makefile
		plugins/puredata/Makefile
		swig/Makefile
	])

echo
echo "**************************************************************"
echo "Summary:"
if test "${FFTW3F_SUPPORT}" = "1"; then
  echo "Fftw3:                   yes (fftw3f)"
else
if test "${FFTW3_SUPPORT}" = "1"; then
  echo "Fftw3:                   yes"
else
  echo "Fftw3f:                  no"
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
else
  echo "Ladcca:                  no"
fi
echo "**************************************************************"
echo Configuration completed successfully. Type \'make\' to build ${PACKAGE}

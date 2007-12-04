pkginclude_HEADERS = aubiocpp.h

lib_LTLIBRARIES = libaubiocpp.la 
libaubiocpp_la_SOURCES = aubiocpp.cpp
AM_CFLAGS = -I$(top_srcdir)/src @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubiocpp_la_LIBADD  = -laubio -L${top_builddir}/src @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubiocpp_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

pkginclude_HEADERS = \
	aubioext.h \
	jackio.h \
	sndfileio.h

lib_LTLIBRARIES = libaubioext.la 
libaubioext_la_SOURCES = aubioext.h \
	jackio.c \
	jackio.h \
	sndfileio.c \
	sndfileio.h 

AM_CFLAGS = -I$(top_srcdir)/src @AUBIO_CFLAGS@ @SNDLIB_CFLAGS@ @JACK_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubioext_la_LIBADD = -laubio -L${top_builddir}/src @SNDLIB_LIBS@ @JACK_LIBS@ @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubioext_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

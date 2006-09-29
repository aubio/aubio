noinst_HEADERS = aubio_priv.h
pkginclude_HEADERS = aubio.h \
	types.h \
	phasevoc.h \
	mathutils.h \
	fft.h \
	sample.h \
	hist.h \
	scale.h \
	resample.h \
	onsetdetection.h \
	tss.h \
	peakpick.h \
	biquad.h \
	pitchdetection.h \
	pitchmcomb.h \
	pitchyin.h \
	pitchschmitt.h \
	pitchfcomb.h \
	pitchyinfft.h \
	beattracking.h \
	onset.h \
	tempo.h \
	filter.h
nodist_pkginclude_HEADERS = config.h

lib_LTLIBRARIES = libaubio.la 
libaubio_la_SOURCES = aubio.h \
	types.h \
	phasevoc.c \
	phasevoc.h \
	mathutils.c \
	mathutils.h \
	fft.c \
	fft.h \
	sample.c \
	sample.h \
	hist.c \
	hist.h \
	scale.c \
	scale.h \
	resample.c \
	resample.h \
	onsetdetection.c \
	onsetdetection.h \
	tss.c \
	tss.h \
	peakpick.c \
	peakpick.h \
	biquad.c \
	biquad.h \
	pitchdetection.c \
	pitchdetection.h \
	pitchmcomb.c \
	pitchmcomb.h \
	pitchyin.c \
	pitchyin.h \
	pitchschmitt.c \
	pitchschmitt.h \
	pitchfcomb.c \
	pitchfcomb.h \
	pitchyinfft.c \
	pitchyinfft.h \
	beattracking.c \
	beattracking.h \
	onset.c \
	onset.h \
	tempo.c \
	tempo.h \
	filter.c \
	filter.h

AM_CFLAGS = @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubio_la_LIBADD = @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubio_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

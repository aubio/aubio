noinst_HEADERS = aubio_priv.h
pkginclude_HEADERS = aubio.h \
	types.h \
	phasevoc.h \
	mathutils.h \
	fft.h \
	sample.h \
	fvec.h \
	cvec.h \
	hist.h \
	scale.h \
	resample.h \
	onset/onsetdetection.h \
	tss.h \
	peakpick.h \
	biquad.h \
	pitchdetection.h \
	pitch/pitchmcomb.h \
	pitch/pitchyin.h \
	pitch/pitchschmitt.h \
	pitch/pitchfcomb.h \
	pitch/pitchyinfft.h \
	beattracking.h \
	onset.h \
	tempo.h \
	filter.h \
	filterbank.h \
	mfcc.h

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
	fvec.c \
	fvec.h \
	cvec.c \
	cvec.h \
	hist.c \
	hist.h \
	scale.c \
	scale.h \
	resample.c \
	resample.h \
	onset/onsetdetection.c \
	onset/onsetdetection.h \
	tss.c \
	tss.h \
	peakpick.c \
	peakpick.h \
	biquad.c \
	biquad.h \
	pitchdetection.c \
	pitchdetection.h \
	pitch/pitchmcomb.c \
	pitch/pitchmcomb.h \
	pitch/pitchyin.c \
	pitch/pitchyin.h \
	pitch/pitchschmitt.c \
	pitch/pitchschmitt.h \
	pitch/pitchfcomb.c \
	pitch/pitchfcomb.h \
	pitch/pitchyinfft.c \
	pitch/pitchyinfft.h \
	beattracking.c \
	beattracking.h \
	onset.c \
	onset.h \
	tempo.c \
	tempo.h \
	filter.c \
	filter.h \
	filterbank.c \
	filterbank.h \
	mfcc.h \
	mfcc.c 

AM_CFLAGS = @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubio_la_LIBADD = @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubio_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

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
	tss.h \
	biquad.h \
	pitch/pitchdetection.h \
	pitch/pitchmcomb.h \
	pitch/pitchyin.h \
	pitch/pitchschmitt.h \
	pitch/pitchfcomb.h \
	pitch/pitchyinfft.h \
	onset/onset.h \
	onset/onsetdetection.h \
	onset/peakpick.h \
	tempo/tempo.h \
	tempo/beattracking.h \
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
	tss.c \
	tss.h \
	biquad.c \
	biquad.h \
	pitch/pitchdetection.c \
	pitch/pitchdetection.h \
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
	onset/onset.c \
	onset/onset.h \
	onset/onsetdetection.c \
	onset/onsetdetection.h \
	onset/peakpick.c \
	onset/peakpick.h \
	tempo/tempo.c \
	tempo/tempo.h \
	tempo/beattracking.c \
	tempo/beattracking.h \
	filter.c \
	filter.h \
	filterbank.c \
	filterbank.h \
	mfcc.h \
	mfcc.c 

AM_CFLAGS = @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubio_la_LIBADD = @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubio_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

noinst_HEADERS = aubio_priv.h
pkginclude_HEADERS = aubio.h \
	types.h \
	fvec.h \
	cvec.h \
	mathutils.h \
	utils/hist.h \
	utils/scale.h \
	temporal/resample.h \
	temporal/biquad.h \
	temporal/filter.h \
	spectral/filterbank.h \
	spectral/mfcc.c \
	spectral/phasevoc.h \
	spectral/fft.h \
	spectral/tss.h \
	spectral/spectral_centroid.h \
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
	tempo/beattracking.h

nodist_pkginclude_HEADERS = config.h

lib_LTLIBRARIES = libaubio.la 
libaubio_la_SOURCES = aubio.h \
	types.h \
	fvec.c \
	fvec.h \
	cvec.c \
	cvec.h \
	mathutils.c \
	mathutils.h \
	utils/hist.c \
	utils/hist.h \
	utils/scale.c \
	utils/scale.h \
	temporal/resample.c \
	temporal/resample.h \
	temporal/biquad.c \
	temporal/biquad.h \
	temporal/filter.c \
	temporal/filter.h \
	spectral/filterbank.c \
	spectral/filterbank.h \
	spectral/mfcc.h \
	spectral/mfcc.c \
	spectral/phasevoc.c \
	spectral/phasevoc.h \
	spectral/fft.c \
	spectral/fft.h \
	spectral/tss.c \
	spectral/tss.h \
	spectral/spectral_centroid.c \
	spectral/spectral_centroid.h \
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
	tempo/beattracking.h

AM_CFLAGS = @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubio_la_LIBADD = @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubio_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

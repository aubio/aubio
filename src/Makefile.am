noinst_HEADERS = \
	aubio_priv.h \
	temporal/filter_priv.h

pkginclude_HEADERS = aubio.h \
	types.h \
	fvec.h \
	lvec.h \
	cvec.h \
	mathutils.h

pkgincludeutilsdir = $(pkgincludedir)/utils
pkgincludetemporaldir = $(pkgincludedir)/temporal
pkgincludespectraldir = $(pkgincludedir)/spectral
pkgincludepitchdir = $(pkgincludedir)/pitch
pkgincludeonsetdir = $(pkgincludedir)/onset
pkgincludetempodir = $(pkgincludedir)/tempo

pkgincludeutils_HEADERS = \
	utils/hist.h \
	utils/scale.h

pkgincludetemporal_HEADERS = \
	temporal/resample.h \
	temporal/biquad.h \
	temporal/filter.h \
	temporal/adesign.h \
	temporal/cdesign.h

pkgincludespectral_HEADERS = \
	spectral/filterbank.h \
	spectral/mfcc.h \
	spectral/phasevoc.h \
	spectral/fft.h \
	spectral/tss.h \
	spectral/spectral_centroid.h

pkgincludepitch_HEADERS = \
	pitch/pitchdetection.h \
	pitch/pitchmcomb.h \
	pitch/pitchyin.h \
	pitch/pitchschmitt.h \
	pitch/pitchfcomb.h \
	pitch/pitchyinfft.h

pkgincludeonset_HEADERS =	\
	onset/onset.h \
	onset/onsetdetection.h \
	onset/peakpick.h

pkgincludetempo_HEADERS =	\
	tempo/tempo.h \
	tempo/beattracking.h

lib_LTLIBRARIES = libaubio.la 
libaubio_la_SOURCES = \
	fvec.c \
	lvec.c \
	cvec.c \
	mathutils.c \
	utils/hist.c \
	utils/scale.c \
	temporal/resample.c \
	temporal/biquad.c \
	temporal/filter.c \
	temporal/adesign.c \
	temporal/cdesign.c \
	spectral/filterbank.c \
	spectral/mfcc.c \
	spectral/phasevoc.c \
	spectral/fft.c \
	spectral/tss.c \
	spectral/spectral_centroid.c \
	pitch/pitchdetection.c \
	pitch/pitchmcomb.c \
	pitch/pitchyin.c \
	pitch/pitchschmitt.c \
	pitch/pitchfcomb.c \
	pitch/pitchyinfft.c \
	onset/onset.c \
	onset/onsetdetection.c \
	onset/peakpick.c \
	tempo/tempo.c \
	tempo/beattracking.c

AM_CFLAGS = @AUBIO_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@
libaubio_la_LIBADD = @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @LTLIBOBJS@
libaubio_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

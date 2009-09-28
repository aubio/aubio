noinst_HEADERS = \
	aubio_priv.h \
	temporal/filter_priv.h

nobase_pkginclude_HEADERS = \
	aubio.h \
	types.h \
	fvec.h \
	lvec.h \
	cvec.h \
	mathutils.h \
	vecutils.h \
	utils/hist.h \
	utils/scale.h \
	temporal/resample.h \
	temporal/biquad.h \
	temporal/filter.h \
	temporal/adesign.h \
	temporal/cdesign.h \
	spectral/filterbank.h \
	spectral/filterbank_mel.h \
	spectral/mfcc.h \
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
libaubio_la_SOURCES = \
	fvec.c \
	lvec.c \
	cvec.c \
	mathutils.c \
	vecutils.c \
	utils/hist.c \
	utils/scale.c \
	temporal/resample.c \
	temporal/biquad.c \
	temporal/filter.c \
	temporal/adesign.c \
	temporal/cdesign.c \
	spectral/filterbank.c \
	spectral/filterbank_mel.c \
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

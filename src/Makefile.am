EXTRA_DIST = aubio_priv.h

pkginclude_HEADERS = aubio.h \
	config.h \
	types.h \
	jackio.h \
	phasevoc.h \
	sndfileio.h \
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
	filter.h \
	midi.h \
	list.h \
	timer.h \
	midi_file.h \
	midi_driver.h \
	midi_event.h \
	midi_track.h \
	midi_player.h \
	midi_parser.h

lib_LTLIBRARIES = libaubio.la 
libaubio_la_SOURCES = aubio.h \
	types.h \
	jackio.c \
	jackio.h \
	phasevoc.c \
	phasevoc.h \
	sndfileio.c \
	sndfileio.h \
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
	filter.c \
	filter.h \
	midi.c \
	midi.h \
	list.c \
	list.h \
	timer.c \
	timer.h \
	midi_alsa_seq.c \
	midi_alsa_raw.c \
	midi_file.c \
	midi_file.h \
	midi_event.c \
	midi_event.h \
	midi_track.c \
	midi_track.h \
	midi_player.c \
	midi_player.h \
	midi_parser.c \
	midi_parser.h \
	midi_driver.h \
	midi_driver.c

AM_CFLAGS = @SNDLIB_CFLAGS@ @JACK_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@ @ALSA_CFLAGS@ @LADCCA_CFLAGS@
libaubio_la_LIBADD = @SNDLIB_LIBS@ @JACK_LIBS@ @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @ALSA_LIBS@ @LADCCA_LIBS@ @LTLIBOBJS@

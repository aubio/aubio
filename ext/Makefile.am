pkginclude_HEADERS = \
	aubioext.h \
	jackio.h \
	sndfileio.h

pkgincludemididir = $(pkgincludedir)/midi
pkgincludemidi_HEADERS = \
	midi/midi.h \
	midi/list.h \
	midi/timer.h \
	midi/midi_file.h \
	midi/midi_driver.h \
	midi/midi_event.h \
	midi/midi_track.h \
	midi/midi_player.h \
	midi/midi_parser.h

lib_LTLIBRARIES = libaubioext.la 
libaubioext_la_SOURCES = aubioext.h \
	jackio.c \
	jackio.h \
	sndfileio.c \
	sndfileio.h \
	midi/midi.c \
	midi/midi.h \
	midi/list.c \
	midi/list.h \
	midi/timer.c \
	midi/timer.h \
	midi/midi_alsa_seq.c \
	midi/midi_alsa_raw.c \
	midi/midi_file.c \
	midi/midi_file.h \
	midi/midi_event.c \
	midi/midi_event.h \
	midi/midi_track.c \
	midi/midi_track.h \
	midi/midi_player.c \
	midi/midi_player.h \
	midi/midi_parser.c \
	midi/midi_parser.h \
	midi/midi_driver.h \
	midi/midi_driver.c

AM_CFLAGS = -I$(top_srcdir)/src @SNDLIB_CFLAGS@ @JACK_CFLAGS@ @FFTWLIB_CFLAGS@ @SAMPLERATE_CFLAGS@ @ALSA_CFLAGS@ @LADCCA_CFLAGS@
libaubioext_la_LIBADD = @SNDLIB_LIBS@ @JACK_LIBS@ @FFTWLIB_LIBS@ @SAMPLERATE_LIBS@ @ALSA_LIBS@ @LADCCA_LIBS@ @LTLIBOBJS@
libaubioext_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

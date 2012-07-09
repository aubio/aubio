# global flags
AM_CFLAGS = -DAUBIO_PREFIX=\"$(prefix)\" -DPACKAGE=\"@PACKAGE@\" -I$(top_srcdir)/src @AUBIO_CFLAGS@ @LASH_CFLAGS@ @SNDFILE_CFLAGS@ @JACK_CFLAGS@
AM_LDFLAGS = -L$(top_builddir)/src -laubio @AUBIO_LIBS@ @LASH_LIBS@

# add your programs to this list
bin_PROGRAMS = \
	aubioonset \
	aubiotrack \
	aubionotes \
	aubiomfcc

noinst_PROGRAMS = \
	aubiopitch \
	aubioquiet

EXTRA_DIST = utils.h sndfileio.h jackio.h

# optionally add sources file for these programs
COMMON_SOURCE_FILES = sndfileio.c jackio.c utils.c
aubioonset_SOURCES = aubioonset.c $(COMMON_SOURCE_FILES)
aubionotes_SOURCES = aubionotes.c $(COMMON_SOURCE_FILES)
aubiotrack_SOURCES = aubiotrack.c $(COMMON_SOURCE_FILES)
aubiopitch_SOURCES = aubiopitch.c $(COMMON_SOURCE_FILES)
aubioquiet_SOURCES = aubioquiet.c $(COMMON_SOURCE_FILES)
aubiomfcc_SOURCES = aubiomfcc.c $(COMMON_SOURCE_FILES)

aubioonset_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@
aubionotes_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@
aubiotrack_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@
aubiopitch_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@
aubioquiet_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@
aubiomfcc_LDADD = @SNDFILE_LIBS@ @JACK_LIBS@

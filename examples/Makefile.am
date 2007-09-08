if COMPILE_TESTS 
SUBDIRS = tests
endif

# global flags
AM_CFLAGS = -DAUBIO_PREFIX=\"$(prefix)\" -I$(top_srcdir)/src -I$(top_srcdir)/ext @AUBIO_CFLAGS@ @LASH_CFLAGS@ @FFTWLIB_CFLAGS@
AM_LDFLAGS = -L$(top_builddir)/src -L$(top_builddir)/ext -laubioext -laubio @LASH_LIBS@

# add your programs to this list
bin_PROGRAMS = \
	aubioonset \
	aubiotrack \
	aubionotes \
	aubiomfcc

noinst_PROGRAMS = \
	aubioquiet

EXTRA_DIST = utils.h

# optionally add sources file for these programs
aubioonset_SOURCES = aubioonset.c utils.c
aubionotes_SOURCES = aubionotes.c utils.c
aubiotrack_SOURCES = aubiotrack.c utils.c
aubioquiet_SOURCES = aubioquiet.c utils.c
aubiomfcc_SOURCES = aubiomfcc.c utils.c

aubioonset_LDADD = @JACK_LIBS@
aubionotes_LDADD = @JACK_LIBS@
aubiotrack_LDADD = @JACK_LIBS@
aubioquiet_LDADD = @JACK_LIBS@
aubiomfcc_LDADD = @JACK_LIBS@

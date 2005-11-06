if COMPILE_TESTS 
SUBDIRS = tests
endif

# global flags
AM_CFLAGS = -DAUBIO_PREFIX=\"$(prefix)\" -I$(srcdir)/../src -I$(srcdir)/../ext @LADCCA_CFLAGS@ @FFTWLIB_CFLAGS@
AM_LDFLAGS = -L../src -L../ext @LADCCA_LIBS@ -laubioext -laubio
#AM_SOURCES = utils.c

# add your programs to this list
bin_PROGRAMS = \
	aubioonset \
	aubiotrack \
	aubioquiet \
	aubionotes

EXTRA_DIST = utils.h

# optionally add sources file for these programs
aubioonset_SOURCES = aubioonset.c utils.c
aubionotes_SOURCES = aubionotes.c utils.c
aubiotrack_SOURCES = aubiotrack.c utils.c
aubioquiet_SOURCES = aubioquiet.c utils.c

aubioonset_LDADD = @JACK_LIBS@
aubionotes_LDADD = @JACK_LIBS@
aubiotrack_LDADD = @JACK_LIBS@
aubioquiet_LDADD = @JACK_LIBS@


#SUBDIRS=onsets

# global flags
AM_CFLAGS = -I../src @LADCCA_CFLAGS@
AM_LDFLAGS = -L../src @LADCCA_LIBS@ -laubio
#AM_SOURCES = utils.c

# add your programs to this list
bin_PROGRAMS = \
	aubioonset \
	aubionotesmedian \
	aubionotes
	
EXTRA_DIST = utils.h

# optionally add sources file for these programs
aubioonset_SOURCES = aubioonset.c utils.c
aubionotes_SOURCES = aubionotes.c utils.c
aubionotesmedian_SOURCES = aubionotesmedian.c utils.c

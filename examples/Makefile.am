
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
	
noinst_PROGRAMS = \
	midiparse \
	midiplay \
	midialsa \
	testforclam
	
man_MANS = aubionotes.1 aubioonset.1
EXTRA_DIST = utils.h $(man_MANS)


# optionally add sources file for these programs
aubioonset_SOURCES = aubioonset.c utils.c
aubionotes_SOURCES = aubionotes.c utils.c
aubionotesmedian_SOURCES = aubionotesmedian.c utils.c
testforclam_SOURCES = testforclam.c utils.c

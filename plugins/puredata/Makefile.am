## Process this file with automake to produce Makefile.in. -*-Makefile-*-
# inspired from Guenter Geiger's Makefile.am for plugin 
# pure-data/externals/plugin~/Makefile.am -- rev 1.1

SUFFIXES = .pd_linux
PDDIR = $(prefix)/lib/pd
pddir = $(PDDIR)

## Make and install the shared library.
pdinstalldir = $(pddir)/extra
pdinstallrefdir = $(pddir)/doc/5.reference
pdinstallexpdir = $(pddir)/doc/aubio

if MINGW
PDEXT = aubio.dll
else
if DARWIN
PDEXT = aubio.pd_darwin
else
PDEXT = aubio.pd_linux
endif
endif

pdinstall_PROGRAMS = $(PDEXT)

ALLSOURCES = \
	aubio_setup.c \
	aubioonset~.c \
	aubiotempo~.c \
	aubiotss~.c \
	aubioquiet~.c \
	aubiopitch~.c

aubio_pd_linux_SOURCES = $(ALLSOURCES)
aubio_pd_darwin_SOURCES = $(ALLSOURCES)
aubio_dll_SOURCES = $(ALLSOURCES)

pdincludedir = $(pddir)/src

AM_CFLAGS  = -I$(top_srcdir)/src -I$(pdincludedir) -DPD -fPIC -DPIC @AUBIO_CFLAGS@
if MINGW
AM_LDFLAGS = -L$(top_builddir)/src/.libs -laubio -export_dynamic -shared 
else
if DARWIN
AM_LDFLAGS = -L$(top_builddir)/src/.libs -laubio -bundle -undefined suppress -flat_namespace
else
AM_LDFLAGS = -L$(top_builddir)/src/.libs -laubio -export_dynamic -shared 
endif
endif


## Install the documentation.

pdinstallref_DATA = \
	help/aubioonset~-help.pd \
	help/aubioquiet~-help.pd \
	help/aubiotempo~-help.pd \
	help/aubiotss~-help.pd \
	help/aubiopitch~-help.pd

pdinstallexp_DATA = \
	examples/onset-cam.pd

EXTRA_DIST = $(pdinstallref_DATA) $(pdinstallexp_DATA)

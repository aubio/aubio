## Process this file with automake to produce Makefile.in. -*-Makefile-*-
# inspired from Guenter Geiger's Makefile.am for plugin 
# pure-data/externals/plugin~/Makefile.am -- rev 1.1

SUFFIXES = .pd_linux
PDDIR = $(prefix)/lib/pd
pddir = $(PDDIR)
#SUBDIRS = ladspa vst win

## Make and install the shared library.
pdinstalldir = $(pddir)/extra
pdinstallrefdir = $(pddir)/doc/5.reference
pdinstallexpdir = $(pddir)/doc/aubio

# Automake won't accept something ending in ".pd_linux" as a library
pdinstall_PROGRAMS = aubio.pd_linux

aubio_pd_linux_SOURCES = \
	aubio_setup.c \
	aubioonset~.c \
	aubiotempo~.c \
	aubiotss~.c \
	aubioquiet~.c \
	aubiopitch~.c

pdincludedir = $(pddir)/src

LIBTOOL=$(SHELL) $(srcdir)/libtoolkludge

INCLUDES = -I$(top_srcdir)/src -I$(pdincludedir)

AM_CFLAGS  = -DPD -fPIC -DPIC
AM_LDFLAGS = -L../../src/.libs -laubio -export_dynamic -shared 

## Install the documentation.

pdinstallref_DATA = \
	help/aubioonset~-help.pd \
	help/aubioquiet~-help.pd \
	help/aubiotempo~-help.pd \
	help/aubiotss~-help.pd \
	help/aubiopitch~-help.pd

pdinstallexp_DATA = \
	examples/onset-cam.pd

## My kludge
noinst_SCRIPTS = libtoolkludge

EXTRA_DIST = $(pdinstallref_DATA) $(pdinstallexp_DATA) $(noinst_SCRIPTS)


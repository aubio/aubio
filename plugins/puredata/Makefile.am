## Process this file with automake to produce Makefile.in. -*-Makefile-*-
# inspired from Guenter Geiger's Makefile.am for plugin 
# pure-data/externals/plugin~/Makefile.am -- rev 1.1

SUFFIXES = .pd_linux
PDDIR = $(prefix)/lib/pd
pddir = $(PDDIR)
#SUBDIRS = ladspa vst win

## Make and install the shared library.
pdinstalldir = $(pddir)/extra

# Automake won't accept something ending in ".pd_linux" as a library
pdinstall_PROGRAMS = aubioonset~.pd_linux

aubioonset__pd_linux_SOURCES = \
	aubioonset~.c

pdincludedir = $(pddir)/src

LIBTOOL=$(SHELL) $(srcdir)/libtoolkludge

INCLUDES = -I$(top_srcdir) -I$(pdincludedir)

AM_CFLAGS = -I$(srcdir)/../../src \
    -DPD -O2 -funroll-loops -fomit-frame-pointer \
    -Wall -W -Wshadow -Wstrict-prototypes -Werror \
    -Wno-unused -Wno-parentheses -Wno-switch
AM_LDFLAGS = -L../../src/.libs -laubio -export_dynamic -shared 

## Install the documentation.

pdinstall_DATA = \
	help-aubioonset~.pd


## My kludge
noinst_SCRIPTS = libtoolkludge

EXTRA_DIST = $(pdinstall_DATA) $(noinst_SCRIPTS)


AM_CFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/ext @LASH_CFLAGS@
AM_LDFLAGS = -L$(top_builddir)/src -L$(top_builddir)/ext -laubioext -laubio @JACK_LIBS@ @LASH_LIBS@

bin_PROGRAMS = \
	test-fft \
	test-mfft \
	test-phasevoc \
	test-phasevoc-jack \
	test-tss

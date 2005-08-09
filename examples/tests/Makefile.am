AM_CFLAGS = -I$(srcdir)/../../src -I$(srcdir)/../../ext @LADCCA_CFLAGS@
AM_LDFLAGS = -L../../src -L../../ext @LADCCA_LIBS@ -laubioext -laubio
LDADD = @JACK_LIBS@

bin_PROGRAMS = \
	test-fft \
	test-mfft \
	test-phasevoc \
	test-phasevoc-jack

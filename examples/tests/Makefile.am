AM_CFLAGS = -I$(top_srcdir)/src
AM_LDFLAGS = -L$(top_builddir)/src -laubio @FFTWLIB_LIBS@

test_phasevoc_jack_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/ext @JACK_CFLAGS@
test_phasevoc_jack_LDADD  = $(AM_LDFLAGS) -laubioext -L$(top_builddir)/ext @JACK_LIBS@ 

bin_PROGRAMS = \
	test-fft \
	test-mfft \
	test-hist \
	test-scale \
	test-sample \
	test-window \
	test-filter \
	test-biquad \
	test-resample \
	test-peakpick \
	test-phasevoc \
	test-phasevoc-jack \
	test-onsetdetection \
	test-pitchyin \
	test-pitchyinfft \
	test-pitchschmitt \
	test-pitchfcomb \
	test-pitchmcomb \
	test-pitchdetection \
	test-beattracking \
	test-onset \
	test-tempo \
	test-tss

run-tests: $(bin_PROGRAMS)
	for i in $(bin_PROGRAMS); do echo $$i; time ./$$i 2>&1 > /dev/null; echo $$?; done

run-valgrind-tests: $(bin_PROGRAMS)
	for i in $(bin_PROGRAMS); do echo $$i; valgrind .libs/lt-$$i 2>&1 | grep ERROR\ SUMMARY -A4; echo $$?; done

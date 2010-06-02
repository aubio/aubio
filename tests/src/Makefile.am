AM_CFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/examples
AM_LDFLAGS = -L$(top_builddir)/src -laubio @FFTWLIB_LIBS@

test_phasevoc_jack_CFLAGS = $(AM_CFLAGS) @JACK_CFLAGS@
test_phasevoc_jack_LDADD  = $(AM_LDFLAGS) @JACK_LIBS@ 

bin_PROGRAMS = \
	test-fft \
	test-hist \
	test-scale \
	test-cvec \
	test-fvec \
	test-window \
	test-filter \
	test-biquad \
	test-resample \
	test-peakpick \
	test-phasevoc \
	test-filterbank \
	test-filterbank_mel \
	test-mfcc \
	test-phasevoc-jack \
	test-onsetdetection \
	test-pitchyin \
	test-pitchyinfft \
	test-pitchschmitt \
	test-pitchfcomb \
	test-pitchmcomb \
	test-pitch \
	test-beattracking \
	test-onset \
	test-tempo \
	test-mathutils \
	test-tss

run-tests: $(bin_PROGRAMS)
	@for i in $(bin_PROGRAMS); do echo -n $$i\ ; ./$$i > /dev/null && echo OK || echo FAILED: $$?; done

run-valgrind-tests: $(bin_PROGRAMS)
	@for i in $(bin_PROGRAMS); do echo $$i; valgrind .libs/lt-$$i 2>&1 | grep ERROR\ SUMMARY -A4; echo $$?; done

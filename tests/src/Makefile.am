AM_CFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/examples
AM_LDFLAGS = -L$(top_builddir)/src -laubio @FFTWLIB_LIBS@

spectral_test_phasevoc_jack_CFLAGS = $(AM_CFLAGS) @JACK_CFLAGS@
spectral_test_phasevoc_jack_LDADD  = $(AM_LDFLAGS) @JACK_LIBS@ 

noinst_PROGRAMS = \
  test-fvec \
  test-cvec \
  test-mathutils \
  test-mathutils-window \
  pitch/test-pitchyin \
  pitch/test-pitchyinfft \
  pitch/test-pitch \
  pitch/test-pitchmcomb \
  pitch/test-pitchfcomb \
  pitch/test-pitchschmitt \
  temporal/test-cweighting \
  temporal/test-resampler \
  temporal/test-filter \
  temporal/test-biquad \
  temporal/test-aweighting \
  onset/test-peakpicker \
  onset/test-onset \
  tempo/test-tempo \
  tempo/test-beattracking \
  spectral/test-fft \
  spectral/test-specdesc \
  spectral/test-phasevoc \
  spectral/test-phasevoc-jack \
  spectral/test-tss \
  spectral/test-filterbank \
  spectral/test-filterbank_mel \
  spectral/test-mfcc \
  utils/test-scale \
  utils/test-hist

run-tests: $(noinst_PROGRAMS)
	@for i in $(noinst_PROGRAMS); do echo -n $$i\ ; ./$$i > /dev/null && echo OK || echo FAILED: $$?; done

run-valgrind-tests: $(noinst_PROGRAMS)
	@for i in $(noinst_PROGRAMS); do echo $$i; valgrind .libs/lt-$$i 2>&1 | grep ERROR\ SUMMARY -A4; echo $$?; done

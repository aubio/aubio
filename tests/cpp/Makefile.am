AM_CXXFLAGS = -I$(top_srcdir)/cpp -I$(top_srcdir)/src
AM_LDFLAGS = -L$(top_builddir)/cpp -laubiocpp -L$(top_builddir)/src -laubio @FFTWLIB_LIBS@

bin_PROGRAMS = \
	test-fvec \
	test-cvec

test_fvec_SOURCES = test-fvec.cpp
test_cvec_SOURCES = test-cvec.cpp

run-tests: $(bin_PROGRAMS)
	@for i in $(bin_PROGRAMS); do echo $$i; ((time ./$$i 2>&1 > /dev/null) 2>&1; echo $$?); done

run-valgrind-tests: $(bin_PROGRAMS)
	@for i in $(bin_PROGRAMS); do echo $$i; valgrind .libs/lt-$$i 2>&1 | grep ERROR\ SUMMARY -A4; echo $$?; done

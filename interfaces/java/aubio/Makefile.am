NOWARN_CFLAGS = -Wno-strict-aliasing

AM_CFLAGS = @AUBIO_CFLAGS@ $(NOWARN_CFLAGS) $(SWCFLAGS) \
	-I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_srcdir)/ext \
	-I/usr/lib/jvm/java-6-sun-1.6.0.04/include \
	-I${prefix}/lib/jvm/java-6-sun-1.6.0.04/include \
	-I/usr/lib/jvm/java-6-sun-1.6.0.04/include/linux \
	-I${prefix}/lib/jvm/java-6-sun-1.6.0.04/include/linux \
	-I/usr/include \
	-I${prefix}/include

SWLDFLAGS = -shared

AUBIO_LDFLAGS = $(SWLDFLAGS) \
	-L$(top_builddir)/ext -laubioext \
	-L$(top_builddir)/src -laubio

pkgjavadir = ${libdir}/java/aubio

pkgjava_LTLIBRARIES = libaubiowrapper.la

libaubiowrapper_la_LDFLAGS = -module -avoid-version $(AUBIO_LDFLAGS) 
libaubiowrapper_la_SOURCES = aubio_wrap.c

aubio_wrap.c:
	$(SWIG) -outdir . -o aubio_wrap.c -java $(top_srcdir)/swig/aubio.i

CLEANFILES = aubio_* SWIGTYPE_* *aubiowrapper* midi_* 

simple_test:
	javac *.java
	LD_LIBRARY_PATH=.libs java main 

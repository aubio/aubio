nobase_pkgpython_PYTHON = \
	__init__.py \
	onsetcompare.py \
	gnuplot.py \
	median.py \
	txtfile.py \
	aubioclass.py \
	task/__init__.py \
	task/params.py \
	task/onset.py \
	task/silence.py \
	task/cut.py \
	task/beat.py \
	task/task.py \
	task/pitch.py \
	task/notes.py \
	bench/__init__.py \
	bench/broadcast.py \
	bench/config.py \
	bench/node.py \
	bench/onset.py \
	plot/__init__.py \
	plot/keyboard.py \
	web/__init__.py \
	web/browser.py \
	web/html.py

nodist_pkgpython_PYTHON = aubiowrapper.py _aubiowrapper.so

_aubiowrapper.so: _aubiowrapper.la

if MINGW
SWLDFLAGS = -L${prefix}/bin -lpython${PYTHON_VERSION}
else
if DARWIN
SWLDFLAGS = -bundle -framework Python
else
SWLDFLAGS = -shared
endif
endif

CLEANFILES = *.pyc *.so *.o aubio_wrap.c aubiowrapper.py 

NOWARN_CFLAGS = -Wno-missing-prototypes -Wno-missing-declarations \
	-Wno-strict-aliasing

AM_CFLAGS = @AUBIO_CFLAGS@ $(NOWARN_CFLAGS) $(SWCFLAGS) \
	-I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_srcdir)/ext \
	-I/usr/include/python${PYTHON_VERSION} \
	-I${prefix}/include/python${PYTHON_VERSION} \
	-I/usr/include \
	-I${prefix}/include

AUBIO_LDFLAGS = $(SWLDFLAGS) \
	-L$(top_builddir)/ext -laubioext \
	-L$(top_builddir)/src -laubio

pkgpyexec_LTLIBRARIES = _aubiowrapper.la

_aubiowrapper_la_LDFLAGS = -module -avoid-version $(AUBIO_LDFLAGS) 
_aubiowrapper_la_SOURCES = aubio_wrap.c

aubio_wrap.c aubiowrapper.py:
	$(SWIG) -outdir . -o aubio_wrap.c -python $(top_srcdir)/swig/aubio.i

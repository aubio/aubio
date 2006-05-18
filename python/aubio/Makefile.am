pkgpython_PYTHON = \
	__init__.py \
	onsetcompare.py \
	gnuplot.py \
	median.py \
	txtfile.py \
	aubioclass.py

pkgpythontaskdir = $(pkgpythondir)/task
pkgpythontask_PYTHON = \
	task/__init__.py \
	task/params.py \
	task/utils.py \
	task/onset.py \
	task/silence.py \
	task/cut.py \
	task/beat.py \
	task/task.py \
	task/pitch.py \
	task/notes.py

pkgpythonbenchdir = $(pkgpythondir)/bench
pkgpythonbench_PYTHON = \
	bench/__init__.py \
	bench/broadcast.py \
	bench/config.py \
	bench/node.py \
	bench/onset.py

pkgpythonplotdir = $(pkgpythondir)/plot
pkgpythonplot_PYTHON = \
	plot/__init__.py \
	plot/keyboard.py

pkgpythonwebdir = $(pkgpythondir)/web
pkgpythonweb_PYTHON = \
	web/__init__.py \
	web/browser.py \
	web/html.py

nodist_pkgpython_PYTHON = aubiowrapper.py _aubiowrapper.so

CLEANFILES = *.pyc *.so *.o aubio_wrap.c aubiowrapper.py 

all: _aubiowrapper.so

# removed -Wall -Wmissing-prototypes -Wmissing-declarations 
CFLAGS = -fPIC -DJACK_SUPPORT -Werror -Wno-char-subscripts \
	-Wno-unknown-pragmas $(SWCFLAGS) \
	-I../../src -I$(srcdir)/../../src -I$(srcdir)/../../ext \
	-I/usr/include/python${PYTHON_VERSION} \
	-I${prefix}/include/python${PYTHON_VERSION} \
	-I/usr/include \
	-I${prefix}/include

LDFLAGS = $(SWLDFLAGS) -L../../ext/.libs -laubioext -L../../src/.libs -laubio

aubio_wrap.c:
	$(SWIG) -outdir . -o aubio_wrap.c -python $(srcdir)/../../swig/aubio.i

_aubiowrapper.so: aubio_wrap.c
	$(CC) $(CFLAGS) -c aubio_wrap.c
	$(CC) $(LDFLAGS) aubio_wrap.o -o _aubiowrapper.so


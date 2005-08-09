pkgpython_PYTHON = __init__.py onsetcompare.py\
	gnuplot.py median.py txtfile.py aubioclass.py

nodist_pkgpython_PYTHON = aubiowrapper.py _aubiowrapper.so

CLEANFILES = *.pyc *.so *.o aubio_wrap.c aubiowrapper.py 

all: _aubiowrapper.so

# removed -Wall -Wmissing-prototypes -Wmissing-declarations 
CFLAGS = -DJACK_SUPPORT -Werror -Wno-char-subscripts \
	-Wno-unknown-pragmas $(SWCFLAGS) \
	-I$(srcdir)/../../src -I$(srcdir)/../../ext \
	-I/usr/include/python${PYTHON_VERSION} \
	-I${prefix}/include/python${PYTHON_VERSION} \
	-I/usr/include \
	-I${prefix}/include

LDFLAGS = $(SWLDFLAGS) -L../../ext/.libs -laubioext -L../../src/.libs -laubio

aubio_wrap.c:
	$(SWIG) -outdir . -o aubio_wrap.c -python ../../swig/aubio.i

_aubiowrapper.so: aubio_wrap.c
	$(CC) $(CFLAGS) -c aubio_wrap.c
	$(CC) $(LDFLAGS) aubio_wrap.o -o _aubiowrapper.so


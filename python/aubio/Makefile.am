pkgpython_PYTHON = __init__.py onsetcompare.py\
	gnuplot.py median.py txtfile.py aubioclass.py
	
nodist_pkgpython_PYTHON = aubiowrapper.py _aubiowrapper.so

CLEANFILES = *.pyc *.so *.o aubio_wrap.c aubiowrapper.py 

all: _aubiowrapper.so

SWIG = swig
# removed -Wall -Wmissing-prototypes -Wmissing-declarations 
SWCFLAGS = -DJACK_SUPPORT -Werror -Wno-char-subscripts -Wno-unknown-pragmas -I/usr/include/python2.3 -I../../src -I../../ext -I/usr/include
SWLDFLAGS = -shared -L../../ext/.libs -laubioext -L../../src/.libs -laubio

aubio_wrap.c:
	$(SWIG) -outdir . -o aubio_wrap.c -python ../../swig/aubio.i

_aubiowrapper.so: aubio_wrap.c
	$(CC)   $(SWCFLAGS) -c aubio_wrap.c
	$(CC)   $(SWLDFLAGS) aubio_wrap.o -o _aubiowrapper.so

EXTRA_DIST = aubio.i

PYTHON_DIR = ../python/aubio
SWIG = swig
#SW_FLAGS = -c++
# removed -Wall -Wmissing-prototypes -Wmissing-declarations 
SW_CFLAGS = -DJACK_SUPPORT -shared -Werror -Wno-char-subscripts -Wno-unknown-pragmas
#SW_CFLAGS = -Wno-missing-prototypes -Wno-missing-declarations
SWINCLUDE = -I/usr/include/python2.3 -I../src -I/usr/include
SWLDFLAGS = -L../src/.libs -laubio


all:
	$(SWIG) $(SW_FLAGS) -outdir $(PYTHON_DIR) -python aubio.i
	$(CC) $(SW_CFLAGS) -c aubio_wrap.c $(SWINCLUDE)
	$(CC) $(SW_CFLAGS) aubio_wrap.o -o $(PYTHON_DIR)/_aubiowrapper.so $(SWLDFLAGS)

install:
	$(mkinstalldirs) $(DESTDIR)$(pkgpythondir)
	$(INSTALL) $(PYTHON_DIR)/_aubiowrapper.so $(DESTDIR)$(pkgpythondir)
	$(INSTALL) -m644 $(PYTHON_DIR)/aubiowrapper.py $(DESTDIR)$(pkgpythondir)

clean:
	rm -f aubio_wrap.{c,o} $(PYTHON_DIR)/{_aubiowrapper.so,aubiowrapper.py,aubiowrapper.pyc}

distclean: clean

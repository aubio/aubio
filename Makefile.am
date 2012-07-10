if DOCBOOKFOUND
DOC = doc
endif

if SWIGFOUND
SWIGDIR = swig
if PYTHONFOUND
PYTHONDIR = python
endif
endif

ACLOCAL_AMFLAGS = -I m4

SUBDIRS = src examples sounds $(PYTHONDIR) $(SWIGDIR) $(DOC) tests
EXTRA_DIST = bootstrap VERSION

docs:
	cd doc && make update-docs && cd ..

doc-devel:
	cd doc && make doc-devel && cd ..

doc-user:
	cd doc && make doc-user && cd ..

doc-examples:
	cd doc && make doc-examples && cd ..

install-pkgconfig:
	$(mkinstalldirs) "$(DESTDIR)$(libdir)/pkgconfig"
	$(INSTALL_DATA) aubio.pc "$(DESTDIR)$(libdir)/pkgconfig/aubio.pc"

uninstall-pkgconfig:
	$(RM) "$(DESTDIR)$(libdir)/pkgconfig/aubio.pc"
	-rmdir "$(DESTDIR)$(libdir)/pkgconfig"

install-data-hook: install-pkgconfig
uninstall-hook: uninstall-pkgconfig

lcov: all
	mkdir -p $(top_builddir)/lcov
	lcov --directory $(top_builddir) --output-file $(top_builddir)/lcov/aubio.info --compat-libtool --zerocounters
	cd tests/python && ./run_all_tests -v || echo "some tests failed"
	lcov --directory $(top_builddir) --output-file $(top_builddir)/lcov/aubio.info --compat-libtool --capture
	genhtml --output-directory $(top_builddir)/lcov $(top_builddir)/lcov/aubio.info 

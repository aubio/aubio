if DOCBOOKFOUND
DOC = doc
endif

if SWIGFOUND
SWIGDIR = swig
if PYTHONFOUND
PYTHONDIR = python
endif
endif

SUBDIRS = src ext examples sounds plugins $(PYTHONDIR) $(SWIGDIR) $(DOC)
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


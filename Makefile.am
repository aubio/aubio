SUBDIRS = src examples doc sounds swig python plugins
EXTRA_DIST = bootstrap VERSION
DISTCLEANFILES = autom4te.cache

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


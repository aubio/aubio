EXTRA_DIST = woodblock.aiff
SUBDIRS = 

install-data-local:
	$(mkinstalldirs) "$(DESTDIR)$(datadir)/sounds/aubio/"
	$(INSTALL_DATA) $(srcdir)/woodblock.aiff "$(DESTDIR)$(datadir)/sounds/aubio"


uninstall-local:
	rm -rf "$(DESTDIR)$(datadir)/sounds/aubio"


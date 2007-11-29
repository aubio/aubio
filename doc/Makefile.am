EXTRA_DIST = devel.cfg examples.cfg user.cfg aubio.css \
	aubionotes.sgml aubioonset.sgml \
	aubiocut.sgml aubiopitch.sgml aubiotrack.sgml \
	aubioonset.1 aubionotes.1 \
	aubiocut.1 aubiopitch.1 aubiotrack.1

man_MANS = aubionotes.1 aubioonset.1 \
	aubiocut.1 aubiopitch.1 aubiotrack.1
CLEANFILES = $(man_MANS)

all: 

update-docs: doc-devel

doc-clean:
	-rm -rf devel

PROJECT_NUMBER = `source $(top_srcdir)/VERSION; echo $$AUBIO_MAJOR_VERSION.$$AUBIO_MINOR_VERSION.$$AUBIO_PATCH_VERSION$$AUBIO_VERSION_STATUS`
maintainer-clean-local: doc-clean

doc-devel: 
	(echo PROJECT_NUMBER = $(PROJECT_NUMBER) ; cat $(srcdir)/devel.cfg)	| doxygen -

doc-user: 
	(echo PROJECT_NUMBER = $(PROJECT_NUMBER) ; cat $(srcdir)/user.cfg)	| doxygen -

doc-examples: 
	(echo PROJECT_NUMBER = $(PROJECT_NUMBER) ; cat $(srcdir)/examples.cfg)	| doxygen -

.sgml.1:
	docbook-to-man $< > $*.1

.sgml.html:
	docbook2html $< && mv index.html $*.html

clean:
	rm -rf devel user examples

distclean-hook: clean

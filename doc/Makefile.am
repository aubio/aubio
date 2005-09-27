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

maintainer-clean-local: doc-clean

doc-devel: 
	doxygen $(srcdir)/devel.cfg

doc-user: 
	doxygen $(srcdir)/user.cfg

doc-examples: 
	doxygen $(srcdir)/examples.cfg

%.sgml: 

%.1:	%.sgml
	docbook-to-man $< > $*.1

clean:
	rm -rf devel user examples

distclean-hook: clean

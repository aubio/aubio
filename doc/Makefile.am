EXTRA_DIST = devel.cfg examples.cfg user.cfg aubio.css \
	aubionotes.sgml aubioonset.sgml \
	aubioonset.1 aubionotes.1

man_MANS = aubionotes.1 aubioonset.1
CLEANFILES = $(man_MANS)

all: 

update-docs: doc-devel

doc-clean:
	-rm -rf devel

maintainer-clean-local: doc-clean

doc-devel: 
	doxygen devel.cfg

doc-user: 
	doxygen user.cfg

doc-examples: 
	doxygen examples.cfg

%.sgml: 

%.1:	%.sgml
	docbook-to-man $< > $*.1

clean:
	rm -rf devel user examples #*.1

distclean-hook: clean

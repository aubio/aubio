#  install the man pages and include in distribution
#man_MANS = fluidsynth.1 
#EXTRA_DIST = $(man_MANS) devel.cfg fluidsynth.1 
EXTRA_DIST = devel.cfg examples.cfg user.cfg aubio.css

#docbook_docs = aubio-devdoc.xml
#DOCBOOK_STYLESHEET ?= http://docbook.sourceforge.net/release/xsl/current/html/chunk.xsl

all: 

# Run "make update-docs" to update developer doc and doxygen reference
#update-docs: xmldocbook doxygen
update-docs: doc-devel

# The new XML DocBook way:
#xmldocbook: $(docbook_docs)
#	-rm -rf html
#	$(mkinstalldirs) html
#	xsltproc -o html/ --catalogs $(DOCBOOK_STYLESHEET) $<

doc-clean:
	-rm -rf devel

maintainer-clean-local: doc-clean

doc-devel: 
	doxygen devel.cfg
	cp aubio.css devel

doc-user: 
	doxygen user.cfg
	cp aubio.css user

doc-examples: 
	doxygen examples.cfg
	cp aubio.css examples
	
man:
	docbook-to-man aubionotes.sgml > aubionotes.1
	docbook-to-man aubioonset.sgml > aubioonset.1

# Update docs for distribution
#dist-hook: 

clean:
	rm -rf devel user examples *.1

distclean: clean

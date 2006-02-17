export BASEDIR=../../../..
export PYTHONPATH=$(BASEDIR)/python
export LD_LIBRARY_PATH=$(BASEDIR)/src/.libs:$(BASEDIR)/ext/.libs

SOURCE = \
	/archives/samples/pitch/isolated

test-aubiopitch:         $(patsubst %, %.aubiopitch, $(SOURCE))

%.aubiopitch: %
	rm -f `basename $<`.aubiocut
	./bench-pitch $< #| tee `basename $<`.aubiocut
	#diff `basename $<`.aubiocut.ref `basename $<`.aubiocut

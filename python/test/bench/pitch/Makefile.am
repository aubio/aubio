export BASEDIR=../../../..
export PYTHONPATH=$(BASEDIR)/python
export LD_LIBRARY_PATH=$(BASEDIR)/src/.libs:$(BASEDIR)/ext/.libs

SOURCE = \
	/var/tmp/pitch \
	$(HOME)/archives/samples/pitch/isolated

test-aubiopitch:         $(patsubst %, %.aubiopitch, $(SOURCE))

%.aubiopitch: %
	rm -f `basename $@`
	./bench-pitch $< #| tee `basename $@`
	-diff `basename $@` `basename $@`

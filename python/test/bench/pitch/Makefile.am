export BASEDIR=../../../..
export PYTHONPATH=$(BASEDIR)/python
export LD_LIBRARY_PATH=$(BASEDIR)/src/.libs:$(BASEDIR)/ext/.libs

SOURCE = \
	$(HOME)/archives/samples/pitch/isolated/piano/011pfnof \
	$(HOME)/archives/samples/pitch/isolated/rhodes/021epnof \
	$(HOME)/archives/samples/pitch/isolated/harpsichord/031hcsim \
	$(HOME)/archives/samples/pitch/isolated/vibraphone/041vihnf

test-aubiopitch:         $(patsubst %, %.aubiopitch, $(SOURCE))

plotpitch:
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(HOME)/archives/samples/pitch/isolated/piano/011pfnof/011pfnof.69.wav -O 011pfnof.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(HOME)/archives/samples/pitch/isolated/rhodes/021epnof/021epnof.69.wav -O 021epnof.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(HOME)/archives/samples/pitch/isolated/harpsichord/031hcsim/031hcsim.69.wav -O 031hcsim.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(HOME)/archives/samples/pitch/isolated/vibraphone/041vihnf/041vihnf.69.wav -O 041vihnf.69.ps

%.aubiopitch: %
	rm -f `basename $@`
	./bench-pitch $< #| tee `basename $@`
	-diff `basename $@` `basename $@`

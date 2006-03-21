export AUBIODIR=../../../..
export PYTHONPATH=$(AUBIODIR)/python
export LD_LIBRARY_PATH=$(AUBIODIR)/src/.libs:$(AUBIODIR)/ext/.libs

CP=cp -a

SOURCE = \
	$(BASEDIR)/isolated/vibraphone \
	$(BASEDIR)/isolated/clavinet \
	$(BASEDIR)/isolated/elecguitar \
	$(BASEDIR)/isolated/piano \
	$(BASEDIR)/isolated/rhodes \
	$(BASEDIR)/isolated


MONOSOURCE = \
	$(BASEDIR)/monophonic/Mirex04/daisy \
	$(BASEDIR)/monophonic/Mirex04/midi \
	$(BASEDIR)/monophonic/Mirex04/jazz \
	$(BASEDIR)/monophonic/Mirex04/pop \
	$(BASEDIR)/monophonic/Mirex04/opera \
	$(BASEDIR)/monophonic/Mirex04

POLYSOURCE = \
	$(BASEDIR)/polyphonic/Mirex04/daisy \
	$(BASEDIR)/polyphonic/Mirex04/midi \
	$(BASEDIR)/polyphonic/Mirex04/jazz \
	$(BASEDIR)/polyphonic/Mirex04/pop \
	$(BASEDIR)/polyphonic/Mirex04/opera \
	$(BASEDIR)/polyphonic/Mirex04

test-aubiopitch-isolated:    $(patsubst %, %.aubiopitch-isolated, $(SOURCE))
test-aubiopitch-monophonic:  $(patsubst %, %.aubiopitch-monophonic, $(MONOSOURCE))
test-aubiopitch-polyphonic:  $(patsubst %, %.aubiopitch-polyphonic, $(POLYSOURCE))

plotpitch:
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(BASEDIR)/isolated/piano/011pfnof/011pfnof.69.wav -O 011pfnof.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(BASEDIR)/isolated/rhodes/021epnof/021epnof.69.wav -O 021epnof.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(BASEDIR)/isolated/harpsichord/031hcsim/031hcsim.69.wav -O 031hcsim.69.ps
	../../../aubiopitch -m mcomb,fcomb,yin,schmitt -i $(BASEDIR)/isolated/vibraphone/041vihnf/041vihnf.69.wav -O 041vihnf.69.ps

timings:
	./bench-timings $(BASEDIR)/polyphonic/Mirex04/opera

%.aubiopitch-isolated: %
	rm -f isolated/`basename $@`
	./bench-pitch $< | tee isolated/`basename $@`
	-diff isolated/`basename $@`.ref isolated/`basename $@`

%.aubiopitch-monophonic: %
	$(CP) monophonic/`basename $@` monophonic/`basename $@`.old
	./bench-pitch-monophonic $< | tee monophonic/`basename $@`
	-diff monophonic/`basename $@`.ref monophonic/`basename $@`
	-diff monophonic/`basename $@`.old monophonic/`basename $@`

%.aubiopitch-polyphonic: %
	$(CP) polyphonic/`basename $@` polyphonic/`basename $@`.old
	./bench-pitch-polyphonic $< | tee polyphonic/`basename $@`
	-diff polyphonic/`basename $@`.ref polyphonic/`basename $@`
	-diff polyphonic/`basename $@`.old polyphonic/`basename $@`

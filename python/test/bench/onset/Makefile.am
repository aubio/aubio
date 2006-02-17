export BASEDIR=../../../..
export PYTHONPATH=$(BASEDIR)/python
export LD_LIBRARY_PATH=$(BASEDIR)/src/.libs:$(BASEDIR)/ext/.libs

SOURCE = \
	/archives/samples/DB/PercussivePhrases/CM18/Samba_Audio \
	/var/tmp/Onset-Mirex2005/poly_pitched \
	/var/tmp/Onset-Mirex2005/solo_bars_and_bells \
	/var/tmp/Onset-Mirex2005/solo_brass \
	/var/tmp/Onset-Mirex2005/solo_drums \
	/var/tmp/Onset-Mirex2005/solo_plucked_strings \
	/var/tmp/Onset-Mirex2005/solo_singing_voice \
	/var/tmp/Onset-Mirex2005/solo_sustained_strings \
	/var/tmp/Onset-Mirex2005/solo_winds \
	/var/tmp/Onset-Mirex2005/complex \
	/var/tmp/Onset-Mirex2005

test-aubiocut:         $(patsubst %, %.aubiocut, $(SOURCE))

%.aubiocut: %
	rm -f `basename $<`.aubiocut
	./bench-onset $< | tee `basename $<`.aubiocut
	diff `basename $<`.aubiocut.ref `basename $<`.aubiocut

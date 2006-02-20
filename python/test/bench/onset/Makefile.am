export BASEDIR=../../../..
export PYTHONPATH=$(BASEDIR)/python
export LD_LIBRARY_PATH=$(BASEDIR)/src/.libs:$(BASEDIR)/ext/.libs

DETAILSOURCE = \
	/var/tmp/Onset-Mirex2005/poly_pitched \
	/var/tmp/Onset-Mirex2005/solo_bars_and_bells \
	/var/tmp/Onset-Mirex2005/solo_drums \
	/var/tmp/Onset-Mirex2005/solo_plucked_strings \
	/var/tmp/Onset-Mirex2005/solo_singing_voice \
	/var/tmp/Onset-Mirex2005/solo_sustained_strings \
	/var/tmp/Onset-Mirex2005/solo_winds \
	/var/tmp/Onset-Mirex2005/complex

SOURCE = /var/tmp/Onset-Mirex2005

TESTSOURCE = \
	/var/tmp/Onset-Mirex2005/solo_bars_and_bells \
	/var/tmp/Onset-Mirex2005/solo_winds \
	/archives/samples/DB/PercussivePhrases/CM18/Samba_Audio

test-aubiocut:            $(patsubst %, %.aubiocut,    $(TESTSOURCE))
test-aubiodelay:          $(patsubst %, %.aubiodelay,  $(TESTSOURCE))
test-aubiowindow:         $(patsubst %, %.aubiowindow, $(TESTSOURCE))

final-aubiocut:           $(patsubst %, %.aubiocut,    $(DETAILSOURCE) $(SOURCE))
final-aubiodelay:         $(patsubst %, %.aubiodelay,  $(SOURCE))
final-aubiowindow:        $(patsubst %, %.aubiowindow, $(SOURCE))

%.aubiocut: %
	rm -f `basename $<`.aubiocut
	./bench-onset $< | tee `basename $<`.aubiocut
	-diff `basename $<`.aubiocut.ref `basename $<`.aubiocut

%.aubiodelay: %
	rm -f `basename $@`
	./bench-delay $< | tee `basename $@`
	-diff `basename $@`.ref `basename $@`

%.aubiowindow: %
	rm -f `basename $@`
	./bench-window $< | tee `basename $@`
	-diff `basename $@`.ref `basename $@`

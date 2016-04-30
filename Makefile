WAFCMD=python waf

SOX=sox

ENABLE_DOUBLE := $(shell [ -z $(HAVE_DOUBLE) ] || echo --enable-double )
TESTSOUNDS := "python/tests/sounds"

all: build

checkwaf:
	@[ -f waf ] || make getwaf

getwaf:
	curl https://waf.io/waf-1.8.20 > waf
	@chmod +x waf

expandwaf:
	@[ -d wafilb ] || rm -fr waflib
	@$(WAFCMD) --help > /dev/null
	@mv .waf*/waflib . && rm -fr .waf*
	@sed '/^#==>$$/,$$d' waf > waf2 && mv waf2 waf
	@chmod +x waf

configure: checkwaf
	$(WAFCMD) configure $(WAFOPTS) $(ENABLE_DOUBLE)

build: configure
	$(WAFCMD) build $(WAFOPTS)

build_python:
	cd python && python ./setup.py generate $(ENABLE_DOUBLE) build

test_python:
	cd python && pip install -v .
	cd python && LD_LIBRARY_PATH=$(PWD)/build/src nose2 --verbose
	cd python && pip uninstall -y -v aubio

test_python_osx:
	cd python && pip install --user -v .
	[ -f build/src/libaubio.[0-9].dylib ] && ( mkdir -p ~/lib && cp -prv build/src/libaubio.4.dylib ~/lib ) || true
	cd python && nose2 --verbose
	cd python && pip uninstall -y -v aubio

clean_python:
	cd python && ./setup.py clean

build_python3:
	cd python && python3 ./setup.py generate $(ENABLE_DOUBLE) build

clean_python3:
	cd python && python3 ./setup.py clean

clean:
	$(WAFCMD) clean

distcheck: checkwaf
	$(WAFCMD) distcheck

help:
	$(WAFCMD) --help

create_test_sounds:
	-[ -z `which $(SOX)` ] && ( echo $(SOX) could not be found) || true
	-mkdir -p $(TESTSOUNDS)
	-$(SOX) -r 44100 -b 16 -n $(TESTSOUNDS)/44100Hz_1f_silence.wav synth 1s silence 0
	-$(SOX) -r  8000 -b 16 -n $(TESTSOUNDS)/8000Hz_30s_silence.wav synth 30 silence 0 vol 0.9
	-$(SOX) -r 32000 -b 16 -n $(TESTSOUNDS)/32000Hz_127f_sine440.wav synth 127 sine 440 vol 0.9
	-$(SOX) -r 22050 -b 16 -n $(TESTSOUNDS)/22050Hz_5s_brownnoise.wav synth 5 brownnoise vol 0.9
	-$(SOX) -r 48000 -b 32 -n $(TESTSOUNDS)/48000Hz_60s_sweep.wav synth 60 sine 0.1-20000 vol 0.9

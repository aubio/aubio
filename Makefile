WAFCMD=python waf
WAFURL=https://waf.io/waf-1.8.22

SOX=sox

ENABLE_DOUBLE := $(shell [ -z $(HAVE_DOUBLE) ] || echo --enable-double )
TESTSOUNDS := python/tests/sounds

all: build

checkwaf:
	@[ -f waf ] || make getwaf

getwaf:
	./scripts/get_waf.sh

expandwaf: getwaf
	[ -d wafilb ] || rm -fr waflib
	$(WAFCMD) --help > /dev/null
	mv .waf*/waflib . && rm -fr .waf*
	sed '/^#==>$$/,$$d' waf > waf2 && mv waf2 waf
	chmod +x waf

cleanwaf:
	rm -rf waf waflib .waf*

configure: checkwaf
	$(WAFCMD) configure $(WAFOPTS) $(ENABLE_DOUBLE)

build: configure
	$(WAFCMD) build $(WAFOPTS)

build_python:
	python ./setup.py build_ext $(ENABLE_DOUBLE)

test_python: export LD_LIBRARY_PATH=$(PWD)/build/src
test_python: local_dylib
	# clean
	-pip uninstall -v -y aubio
	./setup.py clean
	$(WAFCMD) distclean
	# build library
	$(WAFCMD) configure build
	# install python requirements
	pip install -v -r requirements.txt
	# install package
	pip install -v .
	# can clean twice
	./setup.py clean
	./setup.py clean
	# run test with installed package
	./python/tests/run_all_tests --verbose
	nose2 -N 4 #--verbose
	# uninstall package
	pip uninstall -y -v aubio

local_dylib:
	# create links from ~/lib/lib* to build/src/lib*
	[ -f build/src/libaubio.[0-9].dylib ] && ( mkdir -p ~/lib && cp -prv build/src/libaubio.[0-9].dylib ~/lib ) || true

clean_python:
	./setup.py clean

test_pure_python:
	# clean
	-pip uninstall -v -y aubio
	-$(WAFCMD) distclean
	./setup.py clean
	# install python requirements
	pip install --verbose --requirement requirements.txt
	# compile
	CFLAGS=-Os python setup.py build $(ENABLE_DOUBLE)
	# needs recompile, disabled
	#./setup.py test
	# install package
	pip install --verbose .
	# can clean twice
	./setup.py clean
	./setup.py clean
	# run test with installed package
	./python/tests/run_all_tests --verbose
	nose2 -N 4 #--verbose
	# uninstall
	pip uninstall -v -y aubio

test_pure_python_wheel:
	-pip uninstall -v -y aubio
	-rm -rf build/ python/gen/
	-rm -f dist/*.whl
	-pip install -v -r requirements.txt
	-pip install -v wheel
	CFLAGS=-Os python setup.py build_ext $(ENABLE_DOUBLE) bdist_wheel --universal
	wheel install dist/*.whl
	nose2 -N 4
	pip uninstall -v -y aubio

build_python3:
	python3 ./setup.py build_ext $(ENABLE_DOUBLE)

clean_python3:
	python3 ./setup.py clean

clean:
	$(WAFCMD) clean

distclean:
	$(WAFCMD) distclean

distcheck: checkwaf
	$(WAFCMD) distcheck $(WAFOPTS) $(ENABLE_DOUBLE)

help:
	$(WAFCMD) --help

create_test_sounds:
	-[ -z `which $(SOX)` ] && ( echo $(SOX) could not be found) || true
	-mkdir -p $(TESTSOUNDS)
	-$(SOX) -r 44100 -b 16 -n "$(TESTSOUNDS)/44100Hz_1f_silence.wav"          synth 1s   silence 0        dcshift .01
	-$(SOX) -r 22050 -b 16 -n "$(TESTSOUNDS)/22050Hz_5s_brownnoise.wav"   synth 5    brownnoise      vol 0.9
	-$(SOX) -r 32000 -b 16 -n "$(TESTSOUNDS)/32000Hz_127f_sine440.wav"    synth 127s sine 440        vol 0.9
	-$(SOX) -r  8000 -b 16 -n "$(TESTSOUNDS)/8000Hz_30s_silence.wav"      synth 30   silence 0       vol 0.9
	-$(SOX) -r 48000 -b 32 -n "$(TESTSOUNDS)/48000Hz_60s_sweep.wav"       synth 60   sine 100-20000  vol 0.9
	-$(SOX) -r 44100 -b 16 -n "$(TESTSOUNDS)/44100Hz_44100f_sine441.wav"  synth 44100s   sine 441 	vol 0.9
	-$(SOX) -r 44100 -b 16 -n "$(TESTSOUNDS)/44100Hz_100f_sine441.wav"    synth 100s sine 441 	vol 0.9

html:
	cd doc && make html

dist: distclean expandwaf
	$(WAFCMD) dist

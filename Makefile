WAFCMD=python waf
WAFURL=https://waf.io/waf-1.8.22

#WAFOPTS:=
# turn on verbose mode
#WAFOPTS += --verbose
# build wafopts
WAFOPTS += --destdir $(DESTDIR)
# multiple jobs
WAFOPTS += --jobs 4

DESTDIR:=$(PWD)/build/dist
PYDESTDIR:=$(PWD)/build/pydist

# default install locations
PREFIX?=/usr/local
EXEC_PREFIX?=$(PREFIX)
LIBDIR?=$(PREFIX)/lib
INCLUDEDIR?=$(PREFIX)/include
DATAROOTDIR?=$(PREFIX)/share
MANDIR?=$(DATAROOTDIR)/man

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

install:
	# install
	$(WAFCMD) install $(WAFOPTS)
	make list_installed

list_installed:
	find $(DESTDIR) -ls | sed 's|$(DESTDIR)|/«destdir»|'

list_installed_python:
	find $(PYDESTDIR) -ls | sed 's|$(PYDESTDIR)|/«pydestdir»|'

uninstall:
	# uninstall
	$(WAFCMD) uninstall $(WAFOPTS)

delete_install:
	rm -rf $(PWD)/dist/test

build_python:
	# build python-aubio, using locally built libaubio if found
	python ./setup.py build $(ENABLE_DOUBLE)

build_python_extlib:
	# build python-aubio using (locally) installed libaubio
	[ -f $(DESTDIR)/$(INCLUDEDIR)/aubio/aubio.h ]
	[ -d $(DESTDIR)/$(LIBDIR) ]
	[ -f $(DESTDIR)/$(LIBDIR)/pkgconfig/aubio.pc ]
	PKG_CONFIG_PATH=$(DESTDIR)/$(LIBDIR)/pkgconfig \
	CFLAGS="-I$(DESTDIR)/$(INCLUDEDIR) $(CFLAGS)" \
	LDFLAGS="-L$(DESTDIR)/$(LIBDIR) $(CFLAGS)" \
		make build_python
	make list_installed
	cat $(DESTDIR)/$(LIBDIR)/pkgconfig/aubio.pc

deps_python:
	# install or upgrade python requirements
	pip install --verbose --requirement requirements.txt

# use pip or distutils?
#install_python: install_python_with_pip
uninstall_python: uninstall_python_with_pip
install_python: install_python_with_distutils
#uninstall_python: uninstall_python_with_distutils

install_python_with_pip:
	# install package
	pip install --verbose .

uninstall_python_with_pip:
	# uninstall package
	pip uninstall -y -v aubio || make uninstall_python_with_distutils

install_python_with_distutils:
	./setup.py install $(DISTUTILSOPTS)

uninstall_python_with_distutils:
	#./setup.py uninstall
	[ -d $(PYDESTDIR)/$(LIBDIR) ] && echo Warning: did not clean $(PYDESTDIR)/$(LIBDIR) || true

force_uninstall_python:
	# ignore failure if not installed
	-make uninstall_python

local_dylib:
	# DYLD_LIBRARY_PATH is no more on mac os
	# create links from ~/lib/lib* to build/src/lib*
	[ -f $(PWD)/build/src/libaubio.[0-9].dylib ] && ( mkdir -p ~/lib && cp -prv build/src/libaubio.[0-9].dylib ~/lib ) || true

test_python: export LD_LIBRARY_PATH=$(DESTDIR)/$(LIBDIR)
test_python: export PYTHONPATH=$(PYDESTDIR)/$(LIBDIR)
test_python: local_dylib
	# run test with installed package
	./python/tests/run_all_tests --verbose
	# also run with nose, multiple processes
	nose2 -N 4
	# list installed files
	#find $(DESTDIR) -ls | sed 's|$(DESTDIR)||'
	make list_installed_python

clean_python:
	./setup.py clean

check_clean_python:
	# check cleaning a second time works
	make clean_python
	make clean_python

clean: checkwaf
	# optionnaly clean before build
	-$(WAFCMD) clean

check_clean:
	# check cleaning after build works
	$(WAFCMD) clean
	# check cleaning a second time works
	$(WAFCMD) clean

distclean:
	$(WAFCMD) distclean

check_distclean:
	make distclean

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

# build only libaubio, no python-aubio
test_lib_only: clean distclean configure build install
# additionally, clean after a fresh build
test_lib_only_clean: test_lib_only uninstall check_clean check_distclean

# build libaubio, build and test python-aubio against it
test_lib_python: force_uninstall_python deps_python \
	clean_python clean distclean \
	configure build build_python \
	install install_python \
	test_python

test_lib_python_clean: test_lib_python \
	uninstall_python uninstall \
	check_clean_python \
	check_clean \
	check_distclean

# build libaubio, install it, build python-aubio against it
test_lib_install_python: force_uninstall_python deps_python \
	clean_python distclean \
	configure build \
	install \
	build_python_extlib \
	install_python \
	test_python

test_lib_install_python_clean: test_lib_install_python \
	uninstall_python \
	delete_install \
	check_clean_python \
	check_distclean

# build a python-aubio that includes libaubio
test_python_only: force_uninstall_python deps_python \
	clean_python clean distclean \
	build_python \
	install_python \
	test_python

test_python_only_clean: test_python_only \
	uninstall_python \
	check_clean_python


html:
	cd doc && make html

dist: distclean expandwaf
	$(WAFCMD) dist

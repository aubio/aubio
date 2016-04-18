WAFCMD=python waf
all: build

checkwaf:
	@[ -f waf ] || make getwaf

getwaf:
	curl https://waf.io/waf-1.8.14 > waf
	@chmod +x waf

expandwaf:
	@[ -d wafilb ] || rm -fr waflib
	@$(WAFCMD) --help > /dev/null
	@mv .waf*/waflib . && rm -fr .waf*
	@sed '/^#==>$$/,$$d' waf > waf2 && mv waf2 waf
	@chmod +x waf

build: checkwaf
	./waf configure
	./waf build

build_python:
	cd python && ./setup.py build

clean_python:
	cd python && ./setup.py clean

clean:
	$(WAFCMD) clean

distcheck: build
	$(WAFCMD) distcheck

help:
	$(WAFCMD) --help

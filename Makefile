all: build

checkwaf:
	@[ -f waf ] || make getwaf

getwaf:
	curl https://waf.io/waf-1.8.12 > waf
	@[ -d wafilb ] || rm -fr waflib
	@chmod +x waf && ./waf --help > /dev/null
	@mv .waf-*/waflib . && rm -fr .waf-*
	@sed '/^#==>$$/,$$d' waf > waf2 && mv waf2 waf
	@chmod +x waf

build: checkwaf
	./waf configure
	./waf build

clean:
	./waf clean

distcheck: build
	./waf distcheck

help:
	./waf --help

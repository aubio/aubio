#! /bin/sh

set -e
set -x

WAFURL=https://waf.io/waf-2.0.1

( which wget > /dev/null && wget -qO waf $WAFURL ) || ( which curl > /dev/null && curl $WAFURL > waf )

chmod +x waf

#! /bin/sh

set -e
set -x

SCRIPTPATH=`pwd`/$(dirname "$0") 


WAFFILE=waf-1.9.13
WAFURL=https://waf.io/$WAFFILE.tar.bz2

BUILDDIR=$SCRIPTPATH/../.waf-emscripten-dl
mkdir -p $BUILDDIR

cd $BUILDDIR
curl -o $WAFFILE.tar.bz2 $WAFURL
tar xjvf $WAFFILE.tar.bz2
cd $WAFFILE
NOCLIMB=1 ./waf-light --tools=c_emscripten && cp waf $SCRIPTPATH/../   

rm -r $BUILDDIR

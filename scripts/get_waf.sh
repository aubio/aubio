#! /bin/bash

set -e
set -x

WAFVERSION=2.0.1
WAFTARBALL=waf-$WAFVERSION.tar.bz2
WAFURL=https://waf.io/$WAFTARBALL

WAFBUILDDIR=`mktemp -d`

function cleanup () {
  rm -rf $WAFBUILDDIR
}

trap cleanup SIGINT SIGTERM

function buildwaf () {
  pushd $WAFBUILDDIR

  ( which wget > /dev/null && wget -qO $WAFTARBALL $WAFURL ) || ( which curl > /dev/null && curl $WAFURL > $WAFTARBALL )

  tar xf $WAFTARBALL
  pushd waf-$WAFVERSION
  NOCLIMB=1 python waf-light --tools=c_emscripten $*

  popd
  popd

  cp -prv $WAFBUILDDIR/waf-$WAFVERSION/waf $PWD

  chmod +x waf
}

buildwaf

cleanup

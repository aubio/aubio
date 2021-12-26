#! /bin/bash

CREPEURL="https://github.com/marl/crepe/blob/models/model-tiny.h5.bz2?raw=true"
MODELTARGET=crepe-model-tiny.h5
SHA512=91df10316092de0d9c35ae0eaa8f6cceb49fb01f54dc74f9f1196f7f569a3f885242d1abded56c9825180552a602d9e76f2021d7cfb55cfbefb2f84c8a9f4715


function checkmodelsum () {
  ( echo "$SHA512  $MODELTARGET" | shasum -a 512 -c - ) || (echo $MODELTARGET checksum did not match?!; exit 1)
}

function downloadmodel () {
  if command -v wget &> /dev/null
  then
    wget -qO- $1
  else
    curl -Lso- $1
  fi
}

function fetchmodel () {
  downloadmodel $CREPEURL | bunzip2 - > $MODELTARGET
}

( [ -f "$MODELTARGET" ] || fetchmodel ) && checkmodelsum

#! /usr/bin/env bash

# This script sets the environment to execute aubio binaries and python code
# directly from build/ python/build/ without installing libaubio on the system

# Usage: $ source ./scripts/setenv_local.sh

# WARNING: this script will *overwrite* existing (DY)LD_LIBRARY_PATH and
# PYTHONPATH variables.

PYTHON_PLATFORM=`python -c "import pkg_resources, sys; print '%s-%s' % (pkg_resources.get_build_platform(), '.'.join(map(str, sys.version_info[0:2])))"`

AUBIODIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
AUBIOLIB=$AUBIODIR/build/src
AUBIOPYTHON=$AUBIODIR/python/build/lib.$PYTHON_PLATFORM

if [ "$(dirname $PWD)" == "scripts" ]; then
  AUBIODIR=$(basename $PWD)
else
  AUBIODIR=$(basename $PWD)
fi

if [ "$(uname)" == "Darwin" ]; then
  export DYLD_LIBRARY_PATH=$AUBIOLIB
  echo export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH=$AUBIOLIB
  echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
fi

export PYTHONPATH=$AUBIOPYTHON
echo export PYTHONPATH=$PYTHONPATH

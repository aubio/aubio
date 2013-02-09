Python aubio module
===================

This module wraps the aubio library for python using the numpy module.

See the [Python/C API Reference
Manual] (http://docs.python.org/c-api/index.html) and the [Numpy/C API
Reference](http://docs.scipy.org/doc/numpy/reference/c-api.html)

Compiling python aubio on Mac OS X
----------------------------------

You should now be able to build the aubio python module out of the box on a
recent version of OS X (10.8.x). Make sure the variables are correct in the
file `build_osx` before running it:

    $ ./build_osx

Additionally, you can fetch tools such [matplotlib](http://matplotlib.org/) to
use the demo scripts. One easy way to do it is to fetch the fully fledged
[Scipy superpack](http://fonnesbeck.github.com/ScipySuperpack/)

    $ curl -O https://raw.github.com/fonnesbeck/ScipySuperpack/master/install_superpack.sh
    $ sh install_superpack.sh

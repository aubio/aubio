Python aubio module
===================

This module wraps the aubio library for python using the numpy module.

See the [Python/C API Reference
Manual](http://docs.python.org/c-api/index.html) and the [Numpy/C API
Reference](http://docs.scipy.org/doc/numpy/reference/c-api.html)

Compiling python aubio on Mac OS X
----------------------------------

Note: the following URLs to download from are given as examples only, you
should check the corresponding pages for newer versions

Download and install python 2.7 from [python.org](http://www.python.org/)

    $ curl -O http://www.python.org/ftp/python/2.7.3/python-2.7.3-macosx10.6.dmg
    $ open python-2.7.3-macosx10.6.dmg
    # follow the instructions

Download and install [setuptools](http://pypi.python.org/pypi/setuptools)
  
    $ curl -O http://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg
    $ ${SUDO} sh setuptools-0.6c9-py2.4.egg

Download and install [pip](http://www.pip-installer.org/en/latest/installing.html)

    $ curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    $ ${SUDO} python get-pip.py

Download and install [matplotlib](http://matplotlib.sourceforge.net/)

    $ pip install matplotlib

Alternatively, you can fetch the fully fledged [Scipy
superpack](http://fonnesbeck.github.com/ScipySuperpack/)

    $ curl -O https://raw.github.com/fonnesbeck/ScipySuperpack/master/install_superpack.sh
    $ sh install_superpack.sh

You should now be able to build the new python module. make sure the
variables are correct in the file `build_osx` before running it:

    $ ./build_osx

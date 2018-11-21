.. _python-install:

Installing aubio for Python
===========================

aubio is available as a package for Python 2.7 and Python 3. The aubio
extension is written C using the `Python/C`_ and the `Numpy/C`_ APIs.

.. _Python/C: https://docs.python.org/c-api/index.html
.. _Numpy/C: https://docs.scipy.org/doc/numpy/reference/c-api.html

For general documentation on how to install Python packages, see `Installing
Packages`_.

Installing aubio with pip
-------------------------

aubio can be installed from `PyPI`_ using ``pip``:

.. code-block:: console

    $ pip install aubio

See also `Installing from PyPI`_ for general documentation.

.. note::

  aubio is currently a `source only`_ package, so you will need a compiler to
  install it from `PyPI`_. See also `Installing aubio with conda`_ for
  pre-compiled binaries.

.. _PyPI: https://pypi.python.org/pypi/aubio
.. _Installing Packages: https://packaging.python.org/tutorials/installing-packages/
.. _Installing from PyPI: https://packaging.python.org/tutorials/installing-packages/#installing-from-pypi
.. _source only: https://packaging.python.org/tutorials/installing-packages/#source-distributions-vs-wheels

Installing aubio with conda
---------------------------

`Conda packages`_ are available through the `conda-forge`_ channel for Linux,
macOS, and Windows:

.. code-block:: console

    $ conda config --add channels conda-forge
    $ conda install -c conda-forge aubio

.. _Conda packages: https://anaconda.org/conda-forge/aubio
.. _conda-forge: https://conda-forge.org/

.. _py-doubleprecision:

Double precision
----------------

This module can be compiled in double-precision mode, in which case the
default type for floating-point samples will be 64-bit. The default is
single precision mode (32-bit, recommended).

To build the aubio module with double precision, use the option
`--enable-double` of the `build_ext` subcommand:

.. code:: bash

    $ ./setup.py clean
    $ ./setup.py build_ext --enable-double
    $ pip install -v .

**Note**: If linking against `libaubio`, make sure the library was also
compiled in :ref:`doubleprecision` mode.


Checking your installation
--------------------------

Once the python module is installed, its version can be checked with:

.. code-block:: console

    $ python -c "import aubio; print(aubio.version, aubio.float_type)"

The command line `aubio` is also installed:

.. code-block:: console

    $ aubio -h


Python tests
------------

A number of Python tests are provided in the `python tests`_. To run them,
install `nose2`_ and run the script ``python/tests/run_all_tests``:

.. code-block:: console

    $ pip install nose2
    $ ./python/tests/run_all_tests

.. _demo_filter.py: https://github.com/aubio/aubio/blob/master/python/demos/demo_filter.py
.. _python tests: https://github.com/aubio/aubio/blob/master/python/tests
.. _nose2: https://github.com/nose-devs/nose2

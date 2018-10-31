.. _python-install:

Installing aubio for Python
===========================

The aubio extension for Python is available for Python 2.7 and Python 3.

Installing aubio with pip
-------------------------

aubio can now be installed using ``pip``:

.. code-block:: console

    $ pip install aubio

Building the module
-------------------

From ``aubio`` source directory, run the following:

.. code-block:: console

    $ ./setup.py clean
    $ ./setup.py build
    $ sudo ./setup.py install


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

A number of `python tests`_ are provided. To run them, use
``python/tests/run_all_tests``.

.. _demo_filter.py: https://github.com/aubio/aubio/blob/master/python/demos/demo_filter.py
.. _python tests: https://github.com/aubio/aubio/blob/master/python/tests

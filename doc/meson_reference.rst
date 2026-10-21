.. highlight:: bash

.. _meson_quick_reference:

Meson Quick Reference
=====================

This is a quick reference guide for common aubio build tasks using Meson.

Basic Commands
--------------

Setup and Build
~~~~~~~~~~~~~~~

::

    # Initial setup (first time)
    $ meson setup builddir

    # Build
    $ meson compile -C builddir

    # Install
    $ meson install -C builddir

    # All in one
    $ meson setup builddir && meson compile -C builddir && meson install -C builddir

Configuration
~~~~~~~~~~~~~

::

    # Setup with options
    $ meson setup builddir --buildtype=release -Dexamples=true -Dtests=true

    # Reconfigure existing build
    $ meson setup --reconfigure builddir

    # Change options
    $ meson configure builddir -Dexamples=true

    # View current configuration
    $ meson configure builddir

    # Wipe and reconfigure
    $ meson setup --wipe builddir

Cleaning
~~~~~~~~

::

    # Clean build artifacts
    $ meson compile -C builddir --clean

    # Complete clean (remove build directory)
    $ rm -rf builddir

Python Package
--------------

Installation
~~~~~~~~~~~~

::

    # Install Python package
    $ pip install .

    # Editable install for development
    $ pip install -e . --no-build-isolation

    # With uv
    $ uv pip install .
    $ uv pip install -e . --no-build-isolation

Building Wheels
~~~~~~~~~~~~~~~

::

    # Build wheel
    $ pip install build
    $ python -m build

    # With uv
    $ uv build

    # Build for specific Python version
    $ python3.11 -m build

Common Options
--------------

Build Types
~~~~~~~~~~~

* ``--buildtype=debug`` - Debug build with symbols (default)
* ``--buildtype=release`` - Optimized release build
* ``--buildtype=debugoptimized`` - Optimized with debug symbols
* ``--buildtype=minsize`` - Minimized binary size

Aubio-Specific Options
~~~~~~~~~~~~~~~~~~~~~~

::

    # Precision
    -Ddouble=true              # Use double precision (default: false)

    # Components
    -Dexamples=true            # Build example programs (default: false)
    -Dtests=true               # Build test suite (default: false)

    # Dependencies (auto|enabled|disabled)
    -Dfftw3=enabled            # FFTW3 single precision
    -Dfftw3f=enabled           # FFTW3 double precision
    -Dsndfile=enabled          # libsndfile
    -Dsamplerate=enabled       # libsamplerate
    -Djack=enabled             # JACK
    -Davcodec=enabled          # FFmpeg/libav
    -Dvorbis=enabled           # Vorbis
    -Dflac=enabled             # FLAC
    -Drubberband=enabled       # Rubberband

    # Platform-specific
    -Dintelipp=enabled         # Intel IPP (Windows/Linux)
    -Daccelerate=enabled       # Accelerate framework (macOS)
    -Dapple-audio=enabled      # Apple Audio (macOS)

Platform-Specific Builds
------------------------

Windows (MSVC)
~~~~~~~~~~~~~~

::

    # Basic build
    $ meson setup builddir --buildtype=release
    $ meson compile -C builddir

    # With Python
    $ pip install .

macOS
~~~~~

::

    # With Accelerate (automatic)
    $ meson setup builddir --buildtype=release
    $ meson compile -C builddir

    # Install
    $ sudo meson install -C builddir

Linux
~~~~~

::

    # Install dependencies first (Debian/Ubuntu)
    $ sudo apt install libfftw3-dev libsndfile1-dev

    # Build
    $ meson setup builddir --buildtype=release
    $ meson compile -C builddir
    $ sudo meson install -C builddir

Development Workflow
--------------------

Typical Development Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # 1. Initial setup
    $ meson setup builddir

    # 2. Make code changes
    $ vim src/foo.c

    # 3. Rebuild (only changed files)
    $ meson compile -C builddir

    # 4. Test
    $ cd builddir && python -c "import sys; sys.path.insert(0, 'python'); import aubio"

    # 5. Install locally
    $ meson install -C builddir --destdir=/tmp/aubio-install

Python Development
~~~~~~~~~~~~~~~~~~

::

    # Editable install
    $ pip install -e . --no-build-isolation

    # After changes to C code, rebuild
    $ pip install -e . --no-build-isolation --force-reinstall --no-deps

    # Or use meson directly
    $ meson compile -C builddir

Testing
-------

Running Tests
~~~~~~~~~~~~~

::

    # Enable tests
    $ meson configure builddir -Dtests=true

    # Build and run all tests
    $ meson test -C builddir

    # Run specific test
    $ meson test -C builddir test-fvec

    # Verbose output
    $ meson test -C builddir --verbose

    # Run with gdb on failure
    $ meson test -C builddir --gdb

Python Testing
~~~~~~~~~~~~~~

::

    # Run Python tests
    $ cd builddir
    $ python -c "import sys; sys.path.insert(0, 'python'); import aubio.test; aubio.test.run()"

Continuous Integration
----------------------

Basic CI Script
~~~~~~~~~~~~~~~

::

    #!/bin/bash
    set -e

    # Setup
    meson setup builddir --buildtype=release --werror -Dexamples=true -Dtests=true

    # Build
    meson compile -C builddir

    # Test
    meson test -C builddir

    # Install to staging
    DESTDIR=$PWD/install meson install -C builddir

With Python
~~~~~~~~~~~

::

    #!/bin/bash
    set -e

    # Build Python wheel
    pip install build
    python -m build

    # Test wheel
    pip install dist/*.whl
    python -c "import aubio; print(aubio.version)"

Troubleshooting Commands
-------------------------

Verbose Build
~~~~~~~~~~~~~

::

    # Show all commands
    $ meson compile -C builddir --verbose

Introspection
~~~~~~~~~~~~~

::

    # Show build system files
    $ meson introspect builddir --buildsystem-files

    # Show installed files
    $ meson introspect builddir --installed

    # Show dependencies
    $ meson introspect builddir --dependencies

Debugging
~~~~~~~~~

::

    # Reconfigure with debug info
    $ meson setup --wipe builddir --buildtype=debug

    # Run specific test with gdb
    $ meson test -C builddir test-fvec --gdb

Environment Variables
---------------------

Useful environment variables:

::

    # Set compiler
    $ CC=clang meson setup builddir

    # Set custom paths
    $ PKG_CONFIG_PATH=/custom/path:$PKG_CONFIG_PATH meson setup builddir

    # Compiler flags
    $ CFLAGS="-march=native" meson setup builddir

    # Prefix installation
    $ meson setup builddir --prefix=$HOME/.local

Comparison with waf
-------------------

Quick translation table:

+---------------------------+-----------------------------------+
| waf                       | meson                             |
+===========================+===================================+
| ``./waf configure``       | ``meson setup builddir``          |
+---------------------------+-----------------------------------+
| ``./waf build``           | ``meson compile -C builddir``     |
+---------------------------+-----------------------------------+
| ``./waf install``         | ``meson install -C builddir``     |
+---------------------------+-----------------------------------+
| ``./waf clean``           | ``meson compile -C builddir       |
|                           | --clean``                         |
+---------------------------+-----------------------------------+
| ``./waf distclean``       | ``rm -rf builddir``               |
+---------------------------+-----------------------------------+
| ``./waf configure         | ``meson configure builddir``      |
| --help``                  |                                   |
+---------------------------+-----------------------------------+

Common Recipes
--------------

Release Build with All Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    $ meson setup builddir --buildtype=release \
        -Dexamples=true \
        -Dtests=true \
        -Dfftw3=enabled \
        -Dsndfile=enabled \
        -Dsamplerate=enabled
    $ meson compile -C builddir

Development Build
~~~~~~~~~~~~~~~~~

::

    $ meson setup builddir --buildtype=debug
    $ pip install -e . --no-build-isolation

Production Python Package
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    $ meson setup builddir --buildtype=release
    $ python -m build
    $ pip install dist/*.whl

Cross-Platform CI
~~~~~~~~~~~~~~~~~

::

    # Linux
    $ meson setup builddir --buildtype=release --werror
    $ meson compile -C builddir && meson test -C builddir

    # macOS
    $ meson setup builddir --buildtype=release --werror
    $ meson compile -C builddir && meson test -C builddir

    # Windows (PowerShell)
    $ meson setup builddir --buildtype=release
    $ meson compile -C builddir
    $ pip install .

Further Reading
---------------

* `Meson Manual <https://mesonbuild.com/Manual.html>`_
* `meson-python documentation <https://meson-python.readthedocs.io/>`_
* :ref:`building` - Full building guide
* :ref:`develop` - Development guide

.. highlight:: bash

.. _building:

Building aubio
==============

.. note::
    To download a prebuilt version of aubio, see :ref:`download`.

aubio uses `waf`_ to configure, compile, and test the source.
A copy of waf is included in aubio tarball, so all you need is a terminal,
a compiler, and a recent version of python installed.

.. note::
    Make sure you have all the :ref:`requirements` you want before building.

Latest release
--------------

The **latest stable release** can be downloaded from https://aubio.org/download::

        $ curl -O http://aubio.org/pub/aubio-0.4.3.tar.bz2
        $ tar xf aubio-0.4.3.tar.bz2
        $ cd aubio-0.4.3

Git repository
--------------

The **latest git branch** can be obtained with::

        $ git clone git://git.aubio.org/git/aubio
        $ cd aubio

The following command will fetch the correct `waf`_ version (not included in
aubio's git)::

        $ ./scripts/get_waf.sh

.. note::

  Windows users without `Git Bash`_ installed will want to use the following
  commands instead:

  .. code:: bash

        $ curl -fsS -o waf https://waf.io/waf-1.8.22
        $ curl -fsS -o waf.bat https://raw.githubusercontent.com/waf-project/waf/master/utils/waf.bat


Compiling
---------

To compile the C library, examples programs, and tests, run::

        $ ./waf configure

Check out the available options using ``./waf configure --help``. Once
you are done with configuration, you can start building::

        $ ./waf build

To install the freshly built C library and tools, simply run the following
command::

        $ sudo ./waf install

.. note::
  Windows users should simply run ``waf``, without the leading ``./``. For
  instance:

  .. code:: bash

       $ waf configure build

Cleaning
--------

If you wish to uninstall the files installed by the ``install`` command, use
``uninstall``::

        $ sudo ./waf uninstall

To clean the source directory, use the ``clean`` command::

        $ ./waf clean

To also forget the options previously passed to the last ``./waf configure``
invocation, use the ``distclean`` command::

        $ ./waf distclean

.. _waf: https://waf.io/

.. _Git Bash: https://git-for-windows.github.io/

.. toctree::
   :maxdepth: 2

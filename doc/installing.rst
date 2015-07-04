.. highlight:: bash

Installing aubio
================

A number of distributions already include aubio. Check your favorite package
management system, or have a look at the `download page
<http://aubio.org/download>`_.

aubio uses `waf <https://waf.io/>`_ to configure, compile, and test the source.
A copy of ``waf`` is included along aubio, so all you need is a ``terminal``
and a recent ``python`` installed.

Source code
-----------

Check out the `download page <http://aubio.org/download>`_ for more options:
http://aubio.org/download.

The latest stable release can be found at http://aubio.org/pub/::

        $ curl -O http://aubio.org/pub/aubio-0.4.1.tar.bz2
        $ tar xf aubio-0.4.1.tar.bz2
        $ cd aubio-0.4.1

The latest develop branch can be obtained with::

        $ git clone git://git.aubio.org/git/aubio/ aubio-devel
        $ cd aubio-devel
        $ git fetch origin develop:develop
        $ git checkout develop

Compiling
---------

To compile the C library, examples programs, and tests, run::

        $ ./waf configure

Check out the available options using ``./waf configure --help | less``. Once
you are done with configuration, you can start building::

        $ ./waf build

To install the freshly built C library and tools, simply run the following
command::

        $ sudo ./waf install

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

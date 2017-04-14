.. _requirements:

Build options
=============

If built without any external dependencies aubio can be somewhat useful, for
instance to read, process, and write simple wav files.

To support more media input formats and add more features to aubio, you can use
one or all of the following `external libraries`_.

You may also want to know more about the `other options`_ and the `platform
notes`_

The configure script will automatically for these extra libraries. To make sure
the library or feature is used, pass the `--enable-flag` to waf. To disable
this feature, use `--disable-feature`.

To find out more about the build commands, use the `--verbose` option.

External libraries
------------------

External libraries are checked for using ``pkg-config``. Set the
``PKG_CONFIG_PATH`` environment variable if you have them installed in an
unusual location.


.. note::

    If ``pkg-config`` is not found in ``PATH``, the configure step will
    succeed, but none of the external libraries will be used.

libav
.....

  `libav.org <https://libav.org/>`_, open source audio and video processing
  tools.

If all of the following libraries are found, they will be used to compile
``aubio_source_avcodec``. so that ``aubio_source`` will be able to decode audio
from all formats supported by `libav
<https://libav.org/documentation/general.html#Audio-Codecs>`_.

* libavcodec
* libavformat
* libavutil
* libavresample

To enable this option, configure with ``--enable-avcodec``. The build will then
failed if the required libraries are not found. To disable this option,
configure with ``--disable-avcodec``


libsndfile
..........

  `libsndfile <http://www.mega-nerd.com/libsndfile/>`_, a C library for reading
  and writing sampled sound files.

With libsndfile built in, ``aubio_source_sndfile`` will be built in and used by
``aubio_source``.

To enable this option, configure with ``--enable-sndfile``. The build will then
fail if the required library is not found. To disable this option, configure
with ``--disable-sndfile``

libsamplerate
.............

  `libsamplerate <http://www.mega-nerd.com/SRC/>`_, a sample rate converter for
  audio.

With libsamplerate built in, ``aubio_source_sndfile`` will support resampling,
and ``aubio_resample`` will be fully functional.

To enable this option, configure with ``--enable-samplerate``. The build will
then fail if the required library is not found. To disable this option,
configure with ``--disable-samplerate``

libfftw3
........

  `FFTW <http://fftw.org/>`_, a C subroutine for computing the discrete Fourier
  transform

With libfftw3 built in, ``aubio_fft`` will use `FFTW`_ to
compute Fast Fourier Transform (FFT), allowing aubio to compute FFT on length
that are not a power of 2.

To enable this option, configure with ``--enable-fftw3``. The build will
then fail if the required library is not found. To disable this option,
configure with ``--disable-fftw3``

Platform notes
--------------

On all platforms, you will need to have installed:

 - a compiler (gcc, clang, msvc, ...)
 - python (any version >= 2.7, including 3.x)
 - a terminal to run command lines in

Linux
.....

The following `External libraries`_ will be used if found: `libav`_,
`libsamplerate`_, `libsndfile`_, `libfftw3`_.

macOS
.....

The following system frameworks will be used on Mac OS X systems:

  - `Accelerate <https://developer.apple.com/reference/accelerate>`_ to compute
    FFTs and other vectorized operations optimally.

  - `CoreAudio <https://developer.apple.com/reference/coreaudio>`_ and
    `AudioToolbox <https://developer.apple.com/reference/audiotoolbox>`_ to
    decode audio from files and network streams.

.. note::

  To build a fat binary for both ``i386`` and ``x86_64``, use ``./waf configure
  --enable-fat``.

The following `External libraries`_ will also be checked: `libav`_,
`libsamplerate`_, `libsndfile`_, `libfftw3`_.

To build a fat binary on a darwin like system (macOS, tvOS, appleOS, ...)
platforms, configure with ``--enable-fat``.

Windows
.......

To use a specific version of the compiler, ``--msvc_version``. To build for a
specific architecture, use ``--msvc_target``. For instance, to build aubio
for ``x86`` using ``msvc 12.0``, use:

.. code:: bash

    waf configure --msvc_version='msvc 12.0' --msvc_target='x86'


The following `External libraries`_ will be used if found: `libav`_,
`libsamplerate`_, `libsndfile`_, `libfftw3`_.

iOS
...

The following system frameworks will be used on iOS and iOS Simulator.

  - `Accelerate <https://developer.apple.com/reference/accelerate>`_ to compute
    FFTs and other vectorized operations optimally.

  - `CoreAudio <https://developer.apple.com/reference/coreaudio>`_ and
    `AudioToolbox <https://developer.apple.com/reference/audiotoolbox>`_ to
    decode audio from files and network streams.

To build aubio for iOS, configure with ``--with-target-platform=ios``. For the
iOS Simulator, use ``--with-target-platform=iosimulator`` instead.

By default, aubio is built with the following flags on iOS:

.. code:: bash

    CFLAGS="-fembed-bitcode -arch arm64 -arch armv7 -arch armv7s -miphoneos-version-min=6.1"

and on iOS Simulator:

.. code::

    CFLAGS="-arch i386 -arch x86_64 -mios-simulator-version-min=6.1"

Set ``CFLAGS`` and ``LINKFLAGS`` to change these default values, or edit
``wscript`` directly.

Other options
-------------

Some additional options can be passed to the configure step. For the complete
list of options, run:

.. code:: bash

    $ ./waf --help

Here is an example of a custom command:

.. code:: bash

    $ ./waf --verbose configure build install \
                --enable-avcodec --enable-wavread --disable-wavwrite \
                --enable-sndfile --enable-samplerate --enable-docs \
                --destdir $PWD/build/destdir --testcmd="echo %s" \
                --prefix=/opt --libdir=/opt/lib/multiarch \
                --manpagesdir=/opt/share/man  \
                uninstall clean distclean dist distcheck

Double precision
................

To compile aubio in double precision mode, configure with ``--enable-double``.

To compile aubio in single precision mode, use ``--disable-double`` (default).

Disabling the tests
...................

In some case, for instance when cross-compiling, unit tests should not be run.
Option ``--notests`` can be used for this purpose. The tests will not be
executed, but the binaries will be compiled, ensuring that linking against
libaubio works as expected.

.. note::

  The ``--notests`` option should be passed to both ``build`` and ``install``
  targets, otherwise waf will try to run them.

Edit wscript
............

Many of the options are gathered in the file `wscript`. a good starting point
when looking for additional options.

.. _build_docs:

Building the docs
-----------------

If the following command line tools are found, the documentation will be built
built:

 - `doxygen <http://doxygen.org>`_ to build the :ref:`doxygen-documentation`.
 - `txt2man <https://github.com/mvertes/txt2man>`_ to build the :ref:`manpages`
 - `sphinx <http://sphinx-doc.org>`_ to build this document

These tools are searched for in the current ``PATH`` environment variable.
By default, the documentation is built only if the tools are found.

To disable the documentation, configure with ``--disable-docs``. To build with
the documentation, configure with ``--enable-docs``.

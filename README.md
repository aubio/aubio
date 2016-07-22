aubio library
=============

aubio is a library to label music and sounds. It listens to audio signals and
attempts to detect events. For instance, when a drum is hit, at which frequency
is a note, or at what tempo is a rhythmic melody.

Its features include segmenting a sound file before each of its attacks,
performing pitch detection, tapping the beat and producing midi streams from
live audio.

aubio provide several algorithms and routines, including:

  - several onset detection methods
  - different pitch detection methods
  - tempo tracking and beat detection
  - MFCC (mel-frequency cepstrum coefficients)
  - FFT and phase vocoder
  - up/down-sampling
  - digital filters (low pass, high pass, and more)
  - spectral filtering
  - transient/steady-state separation
  - sound file and audio devices read and write access
  - various mathematics utilities for music applications

The name aubio comes from _audio_ with a typo: some errors are likely to be
found in the results.

Python module
-------------

A python module to access the library functions is also provided. Please see
the file [`python/README.md`](python/README.md) for more information on how to
use it.

Examples tools
--------------

A few simple command line tools are included along with the library:

 - `aubioonset` outputs the time stamp of detected note onsets
 - `aubiopitch` attempts to identify a fundamental frequency, or pitch, for
   each frame of the input sound
 - `aubiomfcc` computes Mel-frequency Cepstrum Coefficients
 - `aubiotrack` outputs the time stamp of detected beats
 - `aubionotes` emits midi-like notes, with an onset, a pitch, and a duration
 - `aubioquiet` extracts quiet and loud regions

Additionally, the python module comes with the following script:

 - `aubiocut` slices sound files at onset or beat timestamps

Implementation and Design Basics
--------------------------------

The library is written in C and is optimised for speed and portability.

The C API is designed in the following way:

    aubio_something_t * new_aubio_something (void * args);
    audio_something_do (aubio_something_t * t, void * args);
    smpl_t aubio_something_get_a_parameter (aubio_something_t *t);
    uint_t aubio_something_set_a_parameter (aubio_something_t *t, smpl_t a_parameter);
    void del_aubio_something (aubio_something_t * t);

For performance and real-time operation, no memory allocation or freeing take
place in the `_do` methods. Instead, memory allocation should always take place
in the `new_` methods, whereas free operations are done in the `del_` methods.

The latest version of the documentation can be found at:

  https://aubio.org/documentation

Build Instructions
------------------

A number of distributions already include aubio. Check your favorite package
management system, or have a look at the [download
page](https://aubio.org/download).

aubio uses [waf](https://waf.io/) to configure, compile, and test the source:

    ./waf configure
    ./waf build

If waf is not found in the directory, you can download and install it with:

    make getwaf

aubio compiles on Linux, Mac OS X, Cygwin, and iOS.

Installation
------------

To install aubio library and headers on your system, use:

    sudo ./waf install

To uninstall:

    sudo ./waf uninstall

If you don't have root access to install libaubio on your system, you can use
libaubio without installing libaubio either by setting `LD_LIBRARY_PATH`, or by
copying it to `~/lib`.

On Linux, you should be able to set `LD_LIBRARY_PATH` with:

    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/build/src

On Mac OS X, a copy or a symlink can be made in `~/lib`:

    $ mkdir -p ~/lib
    $ ln -sf $PWD/build/src/libaubio*.dylib ~/lib/

Note on Mac OS X systems older than El Capitan (10.11), the `DYLD_LIBRARY_PATH`
variable can be set as follows:

    $ export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$PWD/build/src

Credits and Publications
------------------------

This library gathers music signal processing algorithms designed at the Centre
for Digital Music and elsewhere. This software project was developed along the
research I did at the Centre for Digital Music, Queen Mary, University of
London. Most of this C code was written by myself, starting from published
papers and existing code. The header files of each algorithm contains brief
descriptions and references to the corresponding papers.

Special thanks go Juan Pablo Bello, Chris Duxbury, Samer Abdallah, Alain de
Cheveigne for their help and publications. Also many thanks to Miguel Ramirez
and Nicolas Wack for their bug fixing.

Substantial informations about the algorithms and their evaluation are gathered
in:

  - Paul Brossier, _[Automatic annotation of musical audio for interactive
    systems](https://aubio.org/phd)_, PhD thesis, Centre for Digital music,
Queen Mary University of London, London, UK, 2006.

Additional results obtained with this software were discussed in the following
papers:

  - P. M. Brossier and J. P. Bello and M. D. Plumbley, [Real-time temporal
    segmentation of note objects in music signals](https://aubio.org/articles/brossier04fastnotes.pdf),
in _Proceedings of the International Computer Music Conference_, 2004, Miami,
Florida, ICMA

  -  P. M. Brossier and J. P. Bello and M. D. Plumbley, [Fast labelling of note
     objects in music signals] (https://aubio.org/articles/brossier04fastnotes.pdf),
in _Proceedings of the International Symposium on Music Information Retrieval_,
2004, Barcelona, Spain


Contact Info and Mailing List
-----------------------------

The home page of this project can be found at: https://aubio.org/

Questions, comments, suggestions, and contributions are welcome. Use the
mailing list: <aubio-user@aubio.org>.

To subscribe to the list, use the mailman form:
http://lists.aubio.org/listinfo/aubio-user/

Alternatively, feel free to contact directly the author.


Copyright and License Information
---------------------------------

Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

aubio is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

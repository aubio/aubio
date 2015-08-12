aubio documentation
===================

aubio is a collection of algorithms and tools to label music and sounds. It
listens to audio signals and attempts to detect events. For instance, when a
drum is hit, at which frequency is a note, or at what tempo is a rhythmic
melody.

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

The name aubio comes from *audio* with a typo: some errors are likely to be
found in the results.

Python module
-------------

A python module to access the library functions is also provided. Please see
the file ``python/README`` for more information on how to use it.

Examples tools
--------------

A few simple command line tools are included along with the library:

 - ``aubioonset`` outputs the time stamp of detected note onsets
 - ``aubiopitch`` attempts to identify a fundamental frequency, or pitch, for
   each frame of the input sound
 - ``aubiomfcc`` computes Mel-frequency Cepstrum Coefficients
 - ``aubiotrack`` outputs the time stamp of detected beats
 - ``aubionotes`` emits midi-like notes, with an onset, a pitch, and a duration
 - ``aubioquiet`` extracts quiet and loud regions

Additionally, the python module comes with the following script:

 - ``aubiocut`` slices sound files at onset or beat timestamps

C API basics
------------

The library is written in C and is optimised for speed and portability.

The C API is designed in the following way:

.. code-block:: c

    aubio_something_t * new_aubio_something(void * args);
    audio_something_do(aubio_something_t * t, void * args);
    smpl_t aubio_something_get_a_parameter(aubio_something_t * t);
    uint_t aubio_something_set_a_parameter(aubio_something_t * t, smpl_t a_parameter);
    void del_aubio_something(aubio_something_t * t);

For performance and real-time operation, no memory allocation or freeing take
place in the ``_do`` methods. Instead, memory allocation should always take place
in the ``new_`` methods, whereas free operations are done in the ``del_`` methods.

.. code-block:: bash

    ./waf configure
    ./waf build
    sudo ./waf install

aubio compiles on Linux, Mac OS X, Cygwin, and iPhone.

Documentation
-------------

- Manual pages: http://aubio.org/documentation
- API documentation: http://aubio.org/doc/latest/

Contribute
----------

- Issue Tracker: https://github.com/piem/aubio/issues
- Source Code: https://github.com/piem/aubio

Contact info
------------

The home page of this project can be found at: http://aubio.org/

Questions, comments, suggestions, and contributions are welcome. Use the
mailing list: <aubio-user@aubio.org>.

To subscribe to the list, use the mailman form:
http://lists.aubio.org/listinfo/aubio-user/

Alternatively, feel free to contact directly the author.


Contents
--------

.. toctree::
   :maxdepth: 1

   installing
   python_module

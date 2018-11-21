aubio
=====

aubio is a collection of tools for music and audio analysis.

This package integrates the aubio library with [NumPy] to provide a set of
efficient tools to process and analyse audio signals, including:

- read audio from any media file, including videos and remote streams
- high quality phase vocoder, spectral filterbanks, and linear filters
- Mel-Frequency Cepstrum Coefficients and standard spectral descriptors
- detection of note attacks (onset)
- pitch tracking (fundamental frequency estimation)
- beat detection and tempo tracking

aubio works with both Python 2 and Python 3.

Built with
----------

The core of aubio is written in C for portability and speed. In addition to
[NumPy], aubio can be optionally built to use one or more of the following
libraries:

- media file reading:

    - [ffmpeg](https://ffmpeg.org) / [avcodec](https://libav.org) to decode and
      read audio from almost any format,
    - [libsndfile](http://www.mega-nerd.com/libsndfile/) to read audio from
      uncompressed sound files,
    - [libsamplerate](http://www.mega-nerd.com/SRC/) to re-sample audio signals
    - [CoreAudio](https://developer.apple.com/reference/coreaudio) to read all
      media files supported natively on macOS, iOS, and tvOS.

- hardware acceleration:

    - [Atlas](http://math-atlas.sourceforge.net/) and
      [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms),
      for accelerated vector and matrix computations,
    - [fftw3](http://fftw.org), to compute fast Fourier Transforms of any size.
    - [Accelerate](https://developer.apple.com/reference/accelerate) for
      hardware accelerated FFT and matrix computations on macOs and iOS,
    - [Intel IPP](https://software.intel.com/en-us/intel-ipp), accelerated
      vector computation and FFT implementation,

Documentation
-------------

- [module documentation][doc_python]
- [installation][doc_python_install]
- [aubio manual][manual]
- [aubio homepage][homepage]

[manual]: https://aubio.org/manual/latest/
[doc_python]: https://aubio.org/manual/latest/python.html
[doc_python_install]: https://aubio.org/manual/latest/python_module.html
[homepage]: https://aubio.org
[NumPy]: https://www.numpy.org

Finding some inspiration
------------------------

Some examples are available in the `python/demos` directory. These scripts are
small programs written in python and using python-aubio.

For instance, `demo_source.py` reads a media file.

    $ ./python/demos/demo_source.py /path/to/sound/sample.wav

and `demo_timestretch_online.py` stretches the original file into a new one:

    $ ./python/demo/demo_timestretch_online.py loop.wav stretched_loop.wav 0.92`

Note: you might need to install additional modules to run some of the demos.
Some demos use [matplotlib](http://matplotlib.org/) to draw plots, others use
[PySoundCard](https://github.com/bastibe/PySoundCard) to play and record
sounds.

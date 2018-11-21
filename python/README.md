Python aubio module
===================

This module wraps the aubio library for Python using the numpy module.

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

Testing the Python module
-------------------------

Python tests are in `python/tests` and use the [nose2 python package][nose2].

To run the all the python tests, use the script:

    $ ./python/tests/run_all_tests

Each test script can also be called one at a time. For instance:

    $ ./python/tests/test_note2midi.py -v

[nose2]: https://github.com/nose-devs/nose2

For more information about how this module works, please refer to the [Python/C
API Reference Manual] (http://docs.python.org/c-api/index.html) and the
[Numpy/C API Reference](http://docs.scipy.org/doc/numpy/reference/c-api.html).

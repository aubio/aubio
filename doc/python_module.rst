.. _python:

Python module
=============

The aubio extension for Python is available for Python 2.7 and Python 3.

Installing aubio with pip
-------------------------

aubio can now be installed using ``pip``:

.. code-block:: bash

    $ pip install aubio

Building the module
-------------------

From ``aubio`` source directory, run the following:

.. code-block:: bash

    $ ./setup.py clean
    $ ./setup.py build
    $ sudo ./setup.py install

Using aubio in python
---------------------

Once you have python-aubio installed, you should be able to run ``python -c
"import aubio; print(aubio.version)"``.

A simple example
................

Here is a :download:`simple script <../python/demos/demo_source_simple.py>`
that reads all the samples from a media file:

.. literalinclude:: ../python/demos/demo_source_simple.py
   :language: python

Filtering an input sound file
.............................

Here is a more complete example, :download:`demo_filter.py
<../python/demos/demo_filter.py>`. This files executes the following:

* read an input media file (``aubio.source``)

* filter it using an `A-weighting <https://en.wikipedia.org/wiki/A-weighting>`_
  filter (``aubio.digital_filter``)

* write result to a new file (``aubio.sink``)

.. literalinclude:: ../python/demos/demo_filter.py
   :language: python

More demos
..........

Check out the `python demos folder`_ for more examples.

Python tests
------------

A number of `python tests`_ are provided. To run them, use
``python/tests/run_all_tests``.

.. _python demos folder: https://github.com/aubio/aubio/blob/master/python/demos
.. _demo_filter.py: https://github.com/aubio/aubio/blob/master/python/demos/demo_filter.py
.. _python tests: https://github.com/aubio/aubio/blob/master/python/tests


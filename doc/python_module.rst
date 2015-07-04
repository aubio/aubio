aubio Python module
===================

Building the module
-------------------

From ``aubio`` source directory, run the following:

.. code-block:: bash

    $ cd python
    $ ./setup.py build
    $ sudo ./setup.py install

Using the module
----------------

To use the python module, simply import aubio:

.. code-block:: python

        #! /usr/bin/env python
        import aubio

        s = aubio.source(sys.argv[1], 0, 256)
        while True:
          samples, read = s()
          print samples
          if read < 256: break

Check out the `python demos for aubio
<https://github.com/piem/aubio/blob/develop/python/demos/>`_ for more examples.


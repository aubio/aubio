.. _develop:

Developping with aubio
======================

Read `Contribute`_ to report issues and request new features.

See `Doxygen documentation`_ for the complete documentation of the C library,
built using `Doxygen <http://www.doxygen.org/>`_.

Below is a brief `Library overview`_.

Library overview
----------------

Here is a brief overview of the C library. See also the `Doxygen
documentation`_ for a more detailed list of available functions.

Vectors and matrix
``````````````````

``fvec_t`` are used to hold vectors of float (``smpl_t``).

.. literalinclude:: ../tests/src/test-fvec.c
   :language: C
   :lines: 7


.. code-block:: C

        // set some elements
        vec->data[511] = 2.;
        vec->data[vec->length-2] = 1.;

Similarly, ``fmat_t`` are used to hold matrix of floats.

.. literalinclude:: ../tests/src/test-fmat.c
   :language: C
   :lines: 9-19

Reading a sound file
````````````````````
In this example, ``aubio_source`` is used to read a media file.

First, create the objects we need.

.. literalinclude:: ../tests/src/io/test-source.c
   :language: C
   :lines: 22-24, 30-32, 34

.. note::
   With ``samplerate = 0``, ``aubio_source`` will be created with the file's
   original samplerate.

Now for the processing loop:

.. literalinclude:: ../tests/src/io/test-source.c
   :language: C
   :lines: 40-44

At the end of the processing loop, clean-up and de-allocate memory:

.. literalinclude:: ../tests/src/io/test-source.c
   :language: C
   :lines: 50-56

See the complete example: :download:`test-source.c
<../tests/src/io/test-source.c>`.

Computing the spectrum
``````````````````````

Now let's create a phase vocoder:

.. literalinclude:: ../tests/src/spectral/test-phasevoc.c
   :language: C
   :lines: 6-11

The processing loop could now look like:

.. literalinclude:: ../tests/src/spectral/test-phasevoc.c
   :language: C
   :lines: 21-35

See the complete example: :download:`test-phasevoc.c
<../tests/src/spectral/test-phasevoc.c>`.

.. _doxygen-documentation:

Doxygen documentation
---------------------

The latest version of the doxygen documentation is available at:

    https://aubio.org/doc/latest

Contribute
----------

Please report any issue and feature request at the `Github issue tracker
<https://github.com/aubio/aubio/issues>`_. Patches and pull-requests welcome!


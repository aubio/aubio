.. _develop:

Developping with aubio
======================

Here is a brief overview of the C library.

For a more detailed list of available functions, see the `API documentation
<https://aubio.org/doc/latest/>`_.

To report issues, ask questions, and request new features, use `Github Issues
<https://github.com/aubio/aubio/issues>`_

Design Basics
-------------

The library is written in C and is optimised for speed and portability.

The C API is designed in the following way:

.. code-block:: C

   // new_ to create an object foobar
   aubio_foobar_t * new_aubio_foobar(void * args);
   // del_ to delete foobar
   void del_aubio_something (aubio_something_t * t);
   // _do to process output = foobar(input)
   audio_something_do (aubio_something_t * t, fvec_t * input, cvec_t * output);
   // _get_param to get foobar.param
   smpl_t aubio_something_get_a_parameter (aubio_something_t * t);
   // _set_param to set foobar.param
   uint_t aubio_something_set_a_parameter (aubio_something_t * t, smpl_t a_parameter);

For performance and real-time operation, no memory allocation or freeing take
place in the `_do` methods. Instead, memory allocation should always take place
in the `new_` methods, whereas free operations are done in the `del_` methods.


Vectors and matrix
------------------

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
--------------------

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

Computing a spectrum
--------------------

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

The latest version of the API documentation is built using `Doxygen
<http://www.doxygen.org/>`_ and is available at:

    https://aubio.org/doc/latest/

Contribute
----------

Please report any issue and feature request at the `Github issue tracker
<https://github.com/aubio/aubio/issues>`_. Patches and pull-requests welcome!

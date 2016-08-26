Welcome
=======

aubio is a collection of algorithms and tools to label and transform music and
sounds. It scans or `listens` to audio signals and attempts to detect musical
events. For instance, when a drum is hit, at which frequency is a note, or at
what tempo is a rhythmic melody.

aubio features include segmenting a sound file before each of its attacks,
performing pitch detection, tapping the beat and producing midi streams from
live audio.

Features
========

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
- sound file read and write access
- various mathematics utilities for music applications

The name aubio comes from *audio* with a typo: some errors are likely to be
found in the results.

Content
=======

.. toctree::
   :maxdepth: 2

   installing
   cli
   python_module

Project pages
=============

* `Project homepage`_: https://aubio.org
* `aubio on github`_: https://github.com/aubio/aubio
* `aubio on pypi`_: https://pypi.python.org/pypi/aubio
* `API documentation`_: https://aubio.org/doc/latest/
* `Mailing lists`_: https://lists.aubio.org

.. _Project homepage: https://aubio.org
.. _aubio on github: https://github.com/aubio/aubio
.. _aubio on pypi: https://pypi.python.org/pypi/aubio
.. _api documentation: https://aubio.org/doc/latest/
.. _Mailing lists: https://lists.aubio.org/

Current status
==============

.. image:: https://travis-ci.org/aubio/aubio.svg?branch=master
   :target: https://travis-ci.org/aubio/aubio
   :alt: Travis build status

.. image:: https://ci.appveyor.com/api/projects/status/f3lhy3a57rkgn5yi?svg=true
   :target: https://ci.appveyor.com/project/piem/aubio/
   :alt: Appveyor build status

.. image:: https://landscape.io/github/aubio/aubio/master/landscape.svg?style=flat
   :target: https://landscape.io/github/aubio/aubio/master
   :alt: Landscape code health

.. image:: http://readthedocs.org/projects/aubio/badge/?version=latest
   :target: http://aubio.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

* `Travis Continuous integration page <https://travis-ci.org/aubio/aubio>`_
* `Appveyor Continuous integration page <https://ci.appveyor.com/project/piem/aubio>`_
* `ReadTheDocs documentation <http://aubio.readthedocs.io/en/latest/>`_

Copyright and License
=====================

Copyright © 2003-2016 Paul Brossier <piem@aubio.org>

aubio is a `free <http://www.debian.org/intro/free>`_ and `open source
<http://www.opensource.org/docs/definition.php>`_ software; **you** can
redistribute it and/or modify it under the terms of the `GNU
<http://www.gnu.org/>`_ `General Public License
<https://www.gnu.org/licenses/gpl.html>`_ as published by the `Free Software
Foundation <https://fsf.org>`_, either version 3 of the License, or (at your
option) any later version.

.. Note:: aubio is not MIT or BSD licensed. Contact the author if you need it
  in your commercial product.

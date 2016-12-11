.. _manpages:

Command line tools
==================

A few simple command line tools are included along with the library.

 - ``aubioonset`` outputs the time stamp of detected note onsets
 - ``aubiopitch`` attempts to identify a fundamental frequency, or pitch, for
   each frame of the input sound
 - ``aubiomfcc`` computes Mel-frequency Cepstrum Coefficients
 - ``aubiotrack`` outputs the time stamp of detected beats
 - ``aubionotes`` emits midi-like notes, with an onset, a pitch, and a duration
 - ``aubioquiet`` extracts quiet and loud regions

Additionally, the python module comes with the following script:

 - ``aubiocut`` slices sound files at onset or beat timestamps


.. toctree::

   cli_features


``aubioonset``
--------------

.. literalinclude:: aubioonset.txt

``aubiopitch``
--------------

.. literalinclude:: aubiopitch.txt

``aubiomfcc``
--------------

.. literalinclude:: aubiomfcc.txt

``aubiotrack``
--------------

.. literalinclude:: aubiotrack.txt

``aubionotes``
--------------

.. literalinclude:: aubionotes.txt

``aubioquiet``
--------------

.. literalinclude:: aubioquiet.txt

``aubiocut``
--------------

.. literalinclude:: aubiocut.txt

.. _xcode-frameworks-label:

Using aubio frameworks in Xcode
-------------------------------

`Binary frameworks`_ are available and ready to use in your XCode project, for
`iOS`_ and `macOS`_.

#. Download the ``framework.zip`` file from the `Download`_ page.

#. Select 'Build Phases' in your project settings

#. Unfold the 'Link Binary with Libraries' list, and add 'AudioToolbox and
   Accelerate frameworks

#. Also add ``aubio.framework`` from https://aubio.org/download.

#. Include the aubio header in your code:

  * in C/C++:

  .. code-block:: c

    #include <aubio/aubio.h>

  * in Obj-C:

  .. code-block:: obj-c

    #import <aubio/aubio.h>

  * in Swift:

  .. code-block:: swift

    import aubio

.. _Binary frameworks: https://aubio.org/download
.. _iOS: https://aubio.org/download#ios
.. _macOS: https://aubio.org/download#osx
.. _Download: https://aubio.org/download

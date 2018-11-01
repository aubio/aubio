"""
This file imports test methods from different testing modules, in this
order:

    - if 'nose2' is found in the list of loaded module, use it
    - otherwise, try using 'pytest'
    - if that also fails, fallback to 'numpy.testing'
"""

import sys

_has_pytest = False
_has_nose2 = False

# if nose2 has already been imported, use it
if 'nose2' in sys.modules:
    from nose2.tools import params, such
    def parametrize(argnames, argvalues):
        return params(*argvalues)
    assert_raises = such.helper.assertRaises
    assert_warns = such.helper.assertWarns
    skipTest = such.helper.skipTest
    _has_nose2 = True

# otherwise, check if we have pytest
if not _has_nose2:
    try:
        import pytest
        parametrize = pytest.mark.parametrize
        assert_raises = pytest.raises
        assert_warns = pytest.warns
        skipTest = pytest.skip
        _has_pytest = True
    except:
        pass

# otherwise fallback on numpy.testing
if not _has_pytest and not _has_nose2:
    from numpy.testing import dec, assert_raises, assert_warns
    from numpy.testing import SkipTest
    parametrize = dec.parametrize
    def skipTest(msg):
        raise SkipTest(msg)

# always use numpy's assert_equal
import numpy
assert_equal = numpy.testing.assert_equal


# add ${src}/python and ${src}/python/aubio/.libs to python path
# so the script is runnable from a compiled source tree.

try:
  from aubio.aubiowrapper import * 
except ImportError:
  try: 
    import os
    import sys
    cur_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(cur_dir,'..','..','python'))
    # waf places
    sys.path.append(os.path.join(cur_dir,'..','..','python','aubio'))
    sys.path.append(os.path.join(cur_dir,'..','..','python','aubio','.libs'))
    # autotools places
    sys.path.append(os.path.join(cur_dir,'..','..','build', 'default', 'swig'))
    sys.path.append(os.path.join(cur_dir,'..','..','build', 'default', 'python','aubio'))
    from aubiowrapper import * 
  except ImportError:
    raise
else:
  raise ImportError, \
    """
    The aubio module could be imported BEFORE adding the source directory to
    your path. Make sure you NO other version of the python aubio module is
    installed on your system.
    """

from template import * 

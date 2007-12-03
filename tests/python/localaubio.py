
# add ${src}/python and ${src}/python/aubio/.libs to python path
# so the script is runnable from a compiled source tree.

try:
  from aubio.aubiowrapper import * 
except ImportError:
  try: 
    import os
    import sys
    cur_dir = os.path.dirname(sys.argv[0])
    sys.path.append(os.path.join(cur_dir,'..','..','python'))
    sys.path.append(os.path.join(cur_dir,'..','..','python','aubio','.libs'))
    from aubio.aubiowrapper import * 
  except ImportError:
    raise
else:
  raise ImportError, "Note: the aubio module could be imported without adding the source directory to your path."

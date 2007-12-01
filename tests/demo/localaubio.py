
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

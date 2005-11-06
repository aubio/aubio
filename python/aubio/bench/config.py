
filefound = 0
try:
        filename = "/etc/aubio-bench.conf"
        execfile(filename)
        filefound = 1
except IOError:
        print "no system wide configuration file found in", filename

try:
        import os
        filename = "%s%s%s" % (os.getenv('HOME'),os.sep,".aubio-bench.conf")
        execfile(filename)
        filefound = 1
except IOError:
        #print "no user configuration file found in", filename
	pass

if filefound == 0:
        import sys
        print "error: no configuration file found at all"
        sys.exit(1)

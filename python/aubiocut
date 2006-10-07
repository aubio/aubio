#! /usr/bin/python

""" this file was written by Paul Brossier 
  it is released under the GNU/GPL license.
"""

import sys
from aubio.task import *

usage = "usage: %s [options] -i soundfile" % sys.argv[0]

def parse_args():
        from optparse import OptionParser
        parser = OptionParser(usage=usage)
        parser.add_option("-i","--input",
                          action="store", dest="filename", 
                          help="input sound file")
        parser.add_option("-m","--mode", 
			  action="store", dest="mode", default='dual', 
                          help="onset detection mode [default=dual] \
                          complexdomain|hfc|phase|specdiff|energy|kl|mkl|dual")
        parser.add_option("-B","--bufsize",
                          action="store", dest="bufsize", default=512, 
                          help="buffer size [default=512]")
        parser.add_option("-H","--hopsize",
                          action="store", dest="hopsize", default=256, 
                          help="overlap size [default=256]")
        parser.add_option("-t","--threshold",
                          action="store", dest="threshold", default=0.3, 
                          help="onset peak picking threshold [default=0.3]")
        parser.add_option("-C","--dcthreshold",
                          action="store", dest="dcthreshold", default=1., 
                          help="onset peak picking DC component [default=1.]")
        parser.add_option("-s","--silence",
                          action="store", dest="silence", default=-70, 
                          help="silence threshold [default=-70]")
        parser.add_option("-M","--mintol",
                          action="store", dest="mintol", default=0.048, 
                          help="minimum inter onset interval [default=0.048]")
        parser.add_option("-D","--delay",
                          action="store", dest="delay",  
                          help="number of seconds to take back [default=system]\
                          default system delay is 3*hopsize/samplerate")
        parser.add_option("-L","--localmin",
                          action="store_true", dest="localmin", default=False, 
                          help="use local minima after peak detection")
        parser.add_option("-c","--cut",
                          action="store_true", dest="cut", default=False,
                          help="cut input sound file at detected labels \
                          best used with option -L")
        parser.add_option("-d","--derivate",
                          action="store_true", dest="derivate", default=False, 
                          help="derivate onset detection function")
        parser.add_option("-S","--silencecut",
                          action="store_true", dest="silencecut", default=False,
                          help="outputs silence locations")
        parser.add_option("-z","--zerocross",
                          action="store", dest="zerothres", default=0.008, 
                          help="zero-crossing threshold for slicing [default=0.00008]")
        # plotting functions
        parser.add_option("-p","--plot",
                          action="store_true", dest="plot", default=False, 
                          help="draw plot")
        parser.add_option("-x","--xsize",
                          action="store", dest="xsize", default=1., 
                          type='float', help="define xsize for plot")
        parser.add_option("-y","--ysize",
                          action="store", dest="ysize", default=1., 
                          type='float', help="define ysize for plot")
        parser.add_option("-f","--function",
                          action="store_true", dest="func", default=False, 
                          help="print detection function")
        parser.add_option("-n","--no-onsets",
                          action="store_true", dest="nplot", default=False, 
                          help="do not plot detected onsets")
        parser.add_option("-O","--outplot",
                          action="store", dest="outplot", default=None, 
                          help="save plot to output.{ps,png}")
        parser.add_option("-F","--spectrogram",
                          action="store_true", dest="spectro", default=False,
                          help="add spectrogram to the plot")
        parser.add_option("-v","--verbose",
                          action="store_true", dest="verbose", default=True,
                          help="make lots of noise [default]")
        parser.add_option("-q","--quiet",
                          action="store_false", dest="verbose", default=True, 
                          help="be quiet")
        # to be implemented
        parser.add_option("-b","--beat",
                          action="store_true", dest="beat", default=False,
                          help="output beat locations")
        (options, args) = parser.parse_args()
        if not options.filename: 
                 print "no file name given\n", usage
                 sys.exit(1)
        return options, args

options, args = parse_args()

filename   = options.filename
params = taskparams()
params.hopsize    = int(options.hopsize)
params.bufsize    = int(options.bufsize)
params.threshold  = float(options.threshold)
params.dcthreshold = float(options.dcthreshold)
params.zerothres  = float(options.zerothres)
params.silence    = float(options.silence)
params.mintol     = float(options.mintol)
params.verbose    = options.verbose
# default take back system delay
if options.delay: params.delay = int(float(options.delay)/params.step)

dotask = taskonset
if options.beat:
	dotask = taskbeat
elif options.silencecut:
	dotask = tasksilence
elif options.plot or options.func: 
	params.storefunc=True
else:              
	params.storefunc=False

lonsets, lofunc = [], []
wplot,oplots = [],[]
modes = options.mode.split(',')
for i in range(len(modes)):
	params.onsetmode = modes[i] 
	filetask = dotask(filename,params=params)
	onsets = filetask.compute_all()

        #lonsets.append(onsets)
	if not options.silencecut:
		ofunc = filetask.ofunc
		lofunc.append(ofunc)

	if options.plot:
		if options.beat: 
			filetask.plot(oplots, onsets)
		else:
			filetask.plot(onsets, ofunc, wplot, oplots, nplot=options.nplot)

	if options.func: 
		for i in ofunc: 
			print i 

if options.outplot:
  extension = options.outplot.split('.')[-1] 
  outplot = '.'.join(options.outplot.split('.')[:-1])
else:
  extension,outplot = None,None
if options.plot: filetask.plotplot(wplot, oplots, outplot=outplot, extension=extension,
  xsize=options.xsize,ysize=options.ysize,spectro=options.spectro)

if options.cut:
        a = taskcut(filename,onsets,params=params)
	a.compute_all()

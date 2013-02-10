"""Copyright (C) 2004 Paul Brossier <piem@altern.org>
print aubio.__LICENSE__ for the terms of use
"""

__LICENSE__ = """\
  Copyright (C) 2004-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.
"""


def audio_to_array(filename):
	import aubio.aubioclass
	from numpy import arange
	hopsize  = 2048
	filei    = aubio.aubioclass.sndfile(filename)
	framestep = 1/(filei.samplerate()+0.)
	channels = filei.channels()
	myvec    = aubio.aubioclass.fvec(hopsize,channels)
	data = []
	readsize = hopsize
	while (readsize==hopsize):
		readsize = filei.read(hopsize,myvec)
		#for i in range(channels):
		i = 0
		curpos = 0
		while (curpos < readsize):
			data.append(myvec.get(curpos,i))
			curpos+=1
	time = arange(len(data))*framestep
	return time,data

def plot_audio(filenames, g, options):
	todraw = len(filenames)
	xorig = 0.
	xratio = 1./todraw
	g('set multiplot;')
	while (len(filenames)):
		time,data = audio_to_array(filenames.pop(0))
		if todraw==1:
			if max(time) < 1.:
				time = [t*1000. for t in time]
				g.xlabel('Time (ms)')
			else:
				g.xlabel('Time (s)')
			g.ylabel('Amplitude')
		curplot = make_audio_plot(time,data)
		g('set size %f,%f;' % (options.xsize*xratio,options.ysize) )
		g('set origin %f,0.;' % (xorig) )
		g('set style data lines; \
			set yrange [-1.:1.]; \
			set xrange [0:%f]' % time[-1]) 
		g.plot(curplot)
		xorig += options.xsize*xratio 
	g('unset multiplot;')

def audio_to_spec(filename,minf = 0, maxf = 0, lowthres = -20., 
		bufsize= 8192, hopsize = 1024):
	from aubioclass import fvec,cvec,pvoc,sndfile
	from math import log10
	filei     = sndfile(filename)
	srate     = float(filei.samplerate())
	framestep = hopsize/srate
	freqstep  = srate/bufsize
	channels  = filei.channels()
	myvec = fvec(hopsize,channels)
	myfft = cvec(bufsize,channels)
	pv    = pvoc(bufsize,hopsize,channels)
	data,time,freq = [],[],[]

	if maxf == 0.: maxf = bufsize/2
	else: maxf = int(maxf/freqstep)
	if minf: minf = int(minf/freqstep)
	else: minf = 0 

	for f in range(minf,maxf):
		freq.append(f*freqstep)
	readsize = hopsize
	frameread = 0
	while (readsize==hopsize):
		readsize = filei.read(hopsize,myvec)
		pv.do(myvec,myfft)
		frame = []
		i = 0 #for i in range(channels):
		curpos = minf 
		while (curpos < maxf):
			frame.append(max(lowthres,20.*log10(myfft.get(curpos,i)**2+0.00001)))
			curpos+=1
		time.append(frameread*framestep)
		data.append(frame)
		frameread += 1
	# crop data if unfinished frames
	if len(data[-1]) != len(data[0]):
		data = data[0:-2]
		time = time[0:-2]
	# verify size consistency
	assert len(data) == len(time)
	assert len(data[0]) == len(freq)
	return data,time,freq

def plot_spec(filename, g, options):
	import Gnuplot
	data,time,freq = audio_to_spec(filename,
    minf=options.minf,maxf=options.maxf,
    bufsize=options.bufsize,hopsize=options.hopsize)
	xorig = 0.
	if max(time) < 1.:
		time = [t*1000. for t in time]
		g.xlabel('Time (ms)')
	else:
		g.xlabel('Time (s)')
	if options.xsize < 0.5 and not options.log and max(time) > 1.:
		freq = [f/1000. for f in freq]
		options.minf /= 1000.
		options.maxf /= 1000.
		g.ylabel('Frequency (kHz)')
	else:
		g.ylabel('Frequency (Hz)')
	g('set pm3d map')
	g('set palette rgbformulae -25,-24,-32')
	g('set cbtics 20')
	#g('set colorbox horizontal')
	g('set xrange [0.:%f]' % time[-1]) 
	if options.log:
		g('set log y')
		g('set yrange [%f:%f]' % (max(10,options.minf),options.maxf))
	else:
		g('set yrange [%f:%f]' % (options.minf,options.maxf))
	g.splot(Gnuplot.GridData(data,time,freq, binary=1))
	#xorig += 1./todraw

def downsample_audio(time,data,maxpoints=10000):
  """ resample audio data to last only maxpoints """
  from numpy import array, resize
  length = len(time)
  downsample = length/maxpoints
  if downsample == 0: downsample = 1
  x = resize(array(time),length)[0:-1:downsample]
  y = resize(array(data),length)[0:-1:downsample]
  return x,y

def make_audio_plot(time,data,maxpoints=10000):
  """ create gnuplot plot from an audio file """
  import Gnuplot, Gnuplot.funcutils
  x,y = downsample_audio(time,data,maxpoints=maxpoints)
  return Gnuplot.Data(x,y,with_='lines')

def make_audio_envelope(time,data,maxpoints=10000):
  """ create gnuplot plot from an audio file """
  from numpy import array
  import Gnuplot, Gnuplot.funcutils
  bufsize = 500
  x = [i.mean() for i in resize(array(time), (len(time)/bufsize,bufsize))] 
  y = [i.mean() for i in resize(array(data), (len(time)/bufsize,bufsize))] 
  x,y = downsample_audio(x,y,maxpoints=maxpoints)
  return Gnuplot.Data(x,y,with_='lines')

def gnuplot_addargs(parser):
  """ add common gnuplot argument to OptParser object """
  parser.add_option("-x","--xsize",
          action="store", dest="xsize", default=1., 
          type='float',help="define xsize for plot")
  parser.add_option("-y","--ysize",
          action="store", dest="ysize", default=1., 
          type='float',help="define ysize for plot")
  parser.add_option("--debug",
          action="store_true", dest="debug", default=False, 
          help="use gnuplot debug mode")
  parser.add_option("--persist",
          action="store_false", dest="persist", default=True, 
          help="do not use gnuplot persistant mode")
  parser.add_option("--lmargin",
          action="store", dest="lmargin", default=None, 
          type='int',help="define left margin for plot")
  parser.add_option("--rmargin",
          action="store", dest="rmargin", default=None, 
          type='int',help="define right margin for plot")
  parser.add_option("--bmargin",
          action="store", dest="bmargin", default=None, 
          type='int',help="define bottom margin for plot")
  parser.add_option("--tmargin",
          action="store", dest="tmargin", default=None, 
          type='int',help="define top margin for plot")
  parser.add_option("-O","--outplot",
          action="store", dest="outplot", default=None, 
          help="save plot to output.{ps,png}")

def gnuplot_create(outplot='',extension='', options=None):
  import Gnuplot
  if options:
    g = Gnuplot.Gnuplot(debug=options.debug, persist=options.persist)
  else:
    g = Gnuplot.Gnuplot(persist=1)
  if not extension or not outplot: return g
  if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
  elif extension == 'eps': ext, extension = '.eps' , 'postscript enhanced'
  elif extension == 'epsc': ext, extension = '.eps' , 'postscript enhanced color'
  elif extension == 'png': ext, extension = '.png', 'png'
  elif extension == 'svg': ext, extension = '.svg', 'svg'
  else: exit("ERR: unknown plot extension")
  g('set terminal %s' % extension)
  if options and options.lmargin: g('set lmargin %i' % options.lmargin)
  if options and options.rmargin: g('set rmargin %i' % options.rmargin)
  if options and options.bmargin: g('set bmargin %i' % options.bmargin)
  if options and options.tmargin: g('set tmargin %i' % options.tmargin)
  if outplot != "stdout":
    g('set output \'%s%s\'' % (outplot,ext))
  if options: g('set size %f,%f' % (options.xsize, options.ysize))
  return g

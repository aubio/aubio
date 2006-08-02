"""Copyright (C) 2004 Paul Brossier <piem@altern.org>
print aubio.__LICENSE__ for the terms of use
"""

__LICENSE__ = """\
	 Copyright (C) 2004 Paul Brossier <piem@altern.org>

	 This program is free software; you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation; either version 2 of the License, or
	 (at your option) any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program; if not, write to the Free Software
	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
"""


__notesheight = 0.25


def audio_to_array(filename):
	import aubio.aubioclass
        import numarray
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
	time = numarray.arange(len(data))*framestep
	return time,data

def plot_audio(filenames, g, start=0, end=None, noaxis=None,xsize=1.,ysize=1.):
	d = []
	todraw = len(filenames)
	xorig = 0.
	xratio = 1./todraw
	#g('set rmargin 7')
	g('set multiplot;')
	while (len(filenames)):
		time,data = audio_to_array(filenames.pop(0))
		if not noaxis and todraw==1:
			if max(time) < 1.:
				time = [t*1000. for t in time]
				g.xlabel('Time (ms)')
			else:
				g.xlabel('Time (s)')
			g.ylabel('Amplitude')
		d.append(make_audio_plot(time,data))
		g('set size %f,%f;' % (xsize*xratio,ysize) )
		g('set origin %f,0.;' % (xorig) )
		g('set style data lines; \
			set yrange [-1.:1.]; \
			set xrange [0:%f]' % time[-1]) 
		g.plot(d.pop(0))
		xorig += xsize*xratio 
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

def plot_spec(filename, g, start=0, end=None, noaxis=None,log=1, minf=0, maxf= 0, xsize = 1., ysize = 1.,bufsize=8192, hopsize=1024):
	import Gnuplot
	data,time,freq = audio_to_spec(filename,minf=minf,maxf=maxf,bufsize=bufsize,hopsize=hopsize)
	xorig = 0.
	if not noaxis:
		if max(time) < 1.:
			time = [t*1000. for t in time]
			g.xlabel('Time (ms)')
		else:
			g.xlabel('Time (s)')
		if xsize < 0.5 and not log and max(time) > 1.:
			freq = [f/1000. for f in freq]
			minf /= 1000.
			maxf /= 1000.
			g.ylabel('Frequency (kHz)')
		else:
			g.ylabel('Frequency (Hz)')
	g('set pm3d map')
	g('set palette rgbformulae -25,-24,-32')
	#g('set lmargin 4')
	g('set cbtics 20')
	#g('set colorbox horizontal')
	g('set xrange [0.:%f]' % time[-1]) 
	if log:
		g('set log y')
		g('set yrange [%f:%f]' % (max(10,minf),maxf))
	else:
		g('set yrange [%f:%f]' % (minf,maxf))
	g.splot(Gnuplot.GridData(data,time,freq, binary=1))
	#xorig += 1./todraw

def downsample_audio(time,data,maxpoints=10000):
	""" resample audio data to last only maxpoints """
	import numarray
        length = len(time)
	downsample = length/maxpoints
        if downsample == 0: downsample = 1
        x = numarray.array(time).resize(length)[0:-1:downsample]
        y = numarray.array(data).resize(length)[0:-1:downsample]
	return x,y

def make_audio_plot(time,data,maxpoints=10000):
	""" create gnuplot plot from an audio file """
	import numarray
	import Gnuplot, Gnuplot.funcutils
        length = len(time)
	downsample = length/maxpoints
        if downsample == 0: downsample = 1
        x = numarray.array(time).resize(length)[0:-1:downsample]
        y = numarray.array(data).resize(length)[0:-1:downsample]
	return Gnuplot.Data(x,y,with='lines')

def gnuplot_init(outplot,debug=0,persist=1):
        # prepare the plot
	import Gnuplot
        g = Gnuplot.Gnuplot(debug=debug, persist=persist)
	if outplot == 'stdout':
                g("set terminal png fontfile 'p052023l.pfb'")
                #g('set output \'%s\'' % outplot)
        elif outplot:
                extension = outplot.split('.')[-1]
                if extension == 'ps': extension = 'postscript'
                g('set terminal %s' % extension)
                g('set output \'%s\'' % outplot)
	return g

def gnuplot_create(outplot='',extension='',debug=0,persist=1, xsize=1., ysize=1.):
	import Gnuplot
        g = Gnuplot.Gnuplot(debug=debug, persist=persist)
	if not extension or not outplot: return g
	if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
	elif extension == 'eps': ext, extension = '.eps' , 'postscript enhanced'
	elif extension == 'epsc': ext, extension = '.eps' , 'postscript enhanced color'
	elif extension == 'png': ext, extension = '.png', 'png'
	elif extension == 'svg': ext, extension = '.svg', 'svg'
	else: exit("ERR: unknown plot extension")
	g('set terminal %s' % extension)
	if outplot != "stdout":
		g('set output \'%s%s\'' % (outplot,ext))
	g('set size %f,%f' % (xsize, ysize))
	return g

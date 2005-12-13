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

def plot_audio(filenames, fileout=None, start=0, end=None, noaxis=None):
	g = gnuplot_init(fileout)
	d = []
	todraw = len(filenames)
	xorig = 0.
	xsize = 1./todraw
	g.gnuplot('set multiplot;')
	while (len(filenames)):
	        time,data = audio_to_array(filenames.pop(0))
                d.append(make_audio_plot(time,data))
		if not noaxis and todraw==1:
			g.xlabel('Time (s)')
			g.ylabel('Amplitude')
		g.gnuplot('set size %f,1.;' % (xsize) )
		g.gnuplot('set origin %f,0.;' % (xorig) )
		g.gnuplot('set style data lines; \
			set yrange [-1.:1.]; \
			set xrange [0:%f]' % time[-1]) 
		g.plot(d.pop(0))
		xorig += 1./todraw
	g.gnuplot('unset multiplot;')

def downsample_audio(time,data,maxpoints=10000):
	""" create gnuplot plot from an audio file """
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


def plot_onsets(filename, onsets, ofunc, samplerate=44100., hopsize=512, outplot=None):
	import Gnuplot, Gnuplot.funcutils
        import aubio.txtfile
        import os.path
        import numarray
	import re
        from aubio.onsetcompare import onset_roc

        d,d2 = [],[]
        maxofunc = 0
        for i in range(len(onsets)):
                if len(onsets[i]) == 0: onsets[i] = [0.];

                # onset detection function 
                downtime = (hopsize/samplerate)*numarray.arange(len(ofunc[i]))
                d.append(Gnuplot.Data(downtime,ofunc[i],with='lines'))
                maxofunc = max(max(ofunc[i]), maxofunc)

        for i in range(len(onsets)):
                # detected onsets
                x1 = (hopsize/samplerate)*numarray.array(onsets[i])
                y1 = maxofunc*numarray.ones(len(onsets[i]))
                d.append(Gnuplot.Data(x1,y1,with='impulses'))
                d2.append(Gnuplot.Data(x1,-y1,with='impulses'))

        # check if datafile exists truth
        datafile = filename.replace('.wav','.txt')
	if datafile == filename: datafile = ""
        if not os.path.isfile(datafile):
                title = "truth file not found"
                t = Gnuplot.Data(0,0,with='impulses') 
        else:
                t_onsets = aubio.txtfile.read_datafile(datafile)
                y2 = maxofunc*numarray.ones(len(t_onsets))
                x2 = numarray.array(t_onsets).resize(len(t_onsets))
                d2.append(Gnuplot.Data(x2,y2,with='impulses'))
                
                tol = 0.050 

                orig, missed, merged, expc, bad, doubled = \
                        onset_roc(x2,x1,tol)
                title = "GD %2.3f%% FP %2.3f%%" % \
                        ((100*float(orig-missed-merged)/(orig)),
                         (100*float(bad+doubled)/(orig)))

        # audio data
        time,data = audio_to_array(filename)
        d2.append(make_audio_plot(time,data))

        # prepare the plot
        g = gnuplot_init(outplot)

        g('set title \'%s %s\'' % (re.sub('.*/','',filename),title))

        g('set multiplot')

        # hack to align left axis
        g('set lmargin 15')

        # plot waveform and onsets
        g('set size 1,0.3')
        g('set origin 0,0.7')
        g('set xrange [0:%f]' % max(time)) 
        g('set yrange [-1:1]') 
        g.ylabel('amplitude')
        g.plot(*d2)
        
        g('unset title')

        # plot onset detection function
        g('set size 1,0.7')
        g('set origin 0,0')
        g('set xrange [0:%f]' % (hopsize/samplerate*len(ofunc[0])))
        g('set yrange [0:%f]' % (maxofunc*1.01))
        g.xlabel('time')
        g.ylabel('onset detection value')
        g.plot(*d)

        g('unset multiplot')


def plot_pitch(filename, pitch, samplerate=44100., hopsize=512, outplot=None):
        import aubio.txtfile
        import os.path
        import numarray
	import Gnuplot
	import re

        d = []
        maxpitch = 100
        for i in range(len(pitch)):
                downtime = (hopsize/samplerate)*numarray.arange(len(pitch[i]))
                d.append(Gnuplot.Data(downtime,pitch[i],with='lines',
                        title=('%d' % i)))
                maxpitch = max(maxpitch,max(pitch[i][:])*1.1)

        # check if ground truth exists
        datafile = filename.replace('.wav','.txt')
	if datafile == filename: datafile = ""
        if not os.path.isfile(datafile):
                title = "truth file not found"
                t = Gnuplot.Data(0,0,with='impulses') 
        else:
                title = "truth file plotting not implemented yet"
                values = aubio.txtfile.read_datafile(datafile)
		if (len(datafile[0])) > 1:
                        time, pitch = [], []
                        for i in range(len(values)):
                                time.append(values[i][0])
                                pitch.append(values[i][1])
                        d.append(Gnuplot.Data(time,pitch,with='lines',
				title='ground truth'))
                
        # audio data
        time,data = audio_to_array(filename)
        f = make_audio_plot(time,data)

	g = gnuplot_init(outplot)
        g('set title \'%s %s\'' % (re.sub('.*/','',filename),title))
        g('set multiplot')
        # hack to align left axis
        g('set lmargin 15')
        # plot waveform and onsets
        g('set size 1,0.3')
        g('set origin 0,0.7')
        g('set xrange [0:%f]' % max(time)) 
        g('set yrange [-1:1]') 
        g.ylabel('amplitude')
        g.plot(f)
        g('unset title')
        # plot onset detection function
        g('set size 1,0.7')
        g('set origin 0,0')
        g('set xrange [0:%f]' % max(time))
        g('set yrange [40:%f]' % maxpitch) 
        g('set key right top')
        g.xlabel('time')
        g.ylabel('frequency (Hz)')
        g.plot(*d)
        g('unset multiplot')

def gnuplot_init(outplot,debug=0,persist=1):
	import Gnuplot
        # prepare the plot
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

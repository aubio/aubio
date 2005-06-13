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

from numarray import *
import Gnuplot, Gnuplot.funcutils

def plotnote(la,title=None) :
	if la[0,:].size() == 3:
	        d = plotnote_withends(la, plot_title=title)
	else: 
	    # scale data if in freq (for REF.txt files)
	    if max(la[:,1] > 128 ):
	        print "scaling frequency data to midi range"
	        la[:,1] /= 6.875
	        la[:,1] = log(la[:,1])/0.6931
	        la[:,1] *= 12
	        la[:,1] -= 3
	    d = plotnote_withoutends(la, plot_title=title)
	return d

def plotnote_multi(lalist,title=None,fileout=None) :
	d=list()
	for i in range(len(lalist)):
	    d.append(plotnote(lalist[i], title=title))
	return d
       

def plotnote_withends(la,plot_title=None) :
	d=[]
	x_widths = array(la[:,1]-la[:,0])/2.
	d.append(Gnuplot.Data(
	        la[:,0]+x_widths,               # x centers
	        la[:,2],                        # y centers
	        x_widths,                       # x errors
	        __notesheight*ones(len(la)),    # y errors
	        title=plot_title,with=('boxxyerrorbars fs 3')))
	return d


def plotnote_withoutends(la,plot_title=None) :
        """ bug: fails drawing last note """
        d=[]
        x_widths = array(la[1:,0]-la[:-1,0])/2;
        d.append(Gnuplot.Data(
                la[:-1,0]+x_widths,             # x centers
                la[:-1,1],                      # y centers
                x_widths,                       # x errors
                __notesheight*ones(len(la)-1),  # y errors
                title=plot_title,with=('boxxyerrorbars fs 3')))
        return d

def plotnote_do(d,fileout=None):
    g = Gnuplot.Gnuplot(debug=1, persist=1)
    g.gnuplot('set style fill solid border 1; \
    set size ratio 1/6; \
    set boxwidth 0.9 relative; \
    set mxtics 2.5; \
    set mytics 2.5; \
    set xtics 5; \
    set ytics 1; \
    set grid xtics ytics mxtics mytics')

    g.xlabel('Time (s)')
    g.ylabel('Midi pitch')
    # do the plot
    #g.gnuplot('set multiplot')
    #for i in d:
    g.plot(d[0])
    #g.gnuplot('set nomultiplot') 
    if fileout != None:
        g.hardcopy(fileout, enhanced=1, color=0)

def audio_to_array(filename):
	import aubio.aubioclass
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

def plot_audio(filenames, fileout=None, start=0, end=None, noaxis=None):
	g = Gnuplot.Gnuplot(debug=1, persist=1)
	d = []
	todraw = len(filenames)
	xorig = 0.
	xsize = 1./todraw
	g.gnuplot('set multiplot;')
	while (len(filenames)):
                d.append(plot_audio_make(filenames.pop(0)))
		if not noaxis and todraw==1:
			g.xlabel('Time (s)')
			g.ylabel('Amplitude')
		g.gnuplot('set size %f,1.;' % (xsize) )
		g.gnuplot('set origin %f,0.;' % (xorig) )
		g.gnuplot('set style data lines; \
			set yrange [-1.:1.]; \
			set xrange [0:%f]' % b[-1]) 
		g.plot(d.pop(0))
		xorig += 1./todraw
	g.gnuplot('unset multiplot;')
	if fileout != None:
		g.hardcopy(fileout, enhanced=1, color=0)

def make_audio_plot(time,data,maxpoints=10000):
	""" create gnuplot plot from an audio file """
        length = len(time)
	downsample = length/maxpoints
        if downsample == 0: downsample = 1
        x = array(time).resize(length)[0:-1:downsample]
        y = array(data).resize(length)[0:-1:downsample]
	return Gnuplot.Data(x,y,with='lines')


def plot_onsets(filename, onsets, ofunc, samplerate=44100., hopsize=512, outplot=None):
        import aubio.txtfile
        import os.path
        import numarray
        from aubio.onsetcompare import onset_roc

        if len(onsets) == 0: onsets = [0.];

        # onset detection function 
        downtime = (hopsize/samplerate)*numarray.arange(len(ofunc))
        d = Gnuplot.Data(downtime,ofunc,with='lines') 

        # detected onsets
        x1 = (hopsize/samplerate)*numarray.array(onsets)
        y1 = max(ofunc)*numarray.ones(len(onsets))
        e = Gnuplot.Data(x1,-y1,with='impulses') 
        e2= Gnuplot.Data(x1,y1,with='impulses') 

        # check if datafile exists truth
        datafile = filename.replace('.wav','.txt')
        if not os.path.isfile(datafile):
                title = "truth file not found"
                t = Gnuplot.Data(0,0,with='impulses') 
        else:
                t_onsets = aubio.txtfile.read_datafile(datafile)
                y2 = max(ofunc)*numarray.ones(len(t_onsets))
                x2 = numarray.array(t_onsets).resize(len(t_onsets))
                t = Gnuplot.Data(x2,y2,with='impulses') 
                
                tol = 0.050 

                orig, missed, merged, expc, bad, doubled = \
                        onset_roc(x2,x1,tol)
                title = "GD %2.3f%% FP %2.3f%%" % \
                        ((100*float(orig-missed-merged)/(orig)),
                         (100*float(bad+doubled)/(orig)))
                #print  orig, missed, merged, expc, bad, doubled
                #print "GD %2.8f\t"        % (100*float(orig-missed-merged)/(orig)),
                #print "FP %2.8f\t"        % (100*float(bad+doubled)/(orig))       , 
                #print "GD-merged %2.8f\t" % (100*float(orig-missed)/(orig))       , 
                #print "FP-pruned %2.8f\t" % (100*float(bad)/(orig))                

        # audio data
        time,data = audio_to_array(filename)
        f = make_audio_plot(time,data)

        # prepare the plot
        g = Gnuplot.Gnuplot(debug=1, persist=1)
        if outplot:
                extension = outplot.split('.')[-1]
                if extension == 'ps': extension = 'postscript'
                g('set terminal %s' % extension)
                g('set output \'%s\'' % outplot)

        g('set title \'%s %s\'' % (filename,title))

        g('set multiplot')

        # hack to align left axis
        g('set lmargin 15')

        # plot waveform and onsets
        g('set size 1,0.3')
        g('set origin 0,0.7')
        g('set xrange [0:%f]' % max(time)) 
        g('set yrange [-1:1]') 
        g.ylabel('amplitude')
        g.plot(f,e,t)
        
        g('unset title')

        # plot onset detection function
        g('set size 1,0.7')
        g('set origin 0,0')
        g('set xrange [0:%f]' % (hopsize/samplerate*len(ofunc)))
        g('set yrange [0:%f]' % (max(ofunc)*1.01))
        g.xlabel('time')
        g.ylabel('onset detection value')
        g.plot(d,e2)

        g('unset multiplot')


def plot_pitch(filename, pitch, samplerate=44100., hopsize=512, outplot=None):
        import aubio.txtfile
        import os.path
        import numarray

        # onset detection function 
        downtime = (hopsize/samplerate)*numarray.arange(len(pitch))
        d = Gnuplot.Data(downtime,pitch,with='lines') 

        # check if datafile exists truth
        datafile = filename.replace('.wav','.txt')
        if not os.path.isfile(datafile):
                title = "truth file not found"
                t = Gnuplot.Data(0,0,with='impulses') 
        else:
                title = "truth file plotting not implemented yet"
                t = Gnuplot.Data(0,0,with='impulses') 
                #times,pitch = aubio.txtfile.read_datafile(datafile)
                #t = Gnuplot.Data(times,pitch,with='lines') 
                
                #orig, missed, merged, expc, bad, doubled = \
                #        onset_roc(x2,x1,tol)
                #title = "GD %2.3f%% FP %2.3f%%" % \
                #        ((100*float(orig-missed-merged)/(orig)),
                #         (100*float(bad+doubled)/(orig)))
                #print  orig, missed, merged, expc, bad, doubled
                #print "GD %2.8f\t"        % (100*float(orig-missed-merged)/(orig)),
                #print "FP %2.8f\t"        % (100*float(bad+doubled)/(orig))       , 
                #print "GD-merged %2.8f\t" % (100*float(orig-missed)/(orig))       , 
                #print "FP-pruned %2.8f\t" % (100*float(bad)/(orig))                

        # audio data
        time,data = audio_to_array(filename)
        f = make_audio_plot(time,data)

        # prepare the plot
        g = Gnuplot.Gnuplot(debug=1, persist=1)
        if outplot:
                extension = outplot.split('.')[-1]
                if extension == 'ps': extension = 'postscript'
                g('set terminal %s' % extension)
                g('set output \'%s\'' % outplot)

        g('set title \'%s %s\'' % (filename,title))

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
        g('set xrange [0:%f]' % (hopsize/samplerate*len(pitch)))
        g('set yrange [0:%f]' % (max(pitch)*1.01))
        g.xlabel('time')
        g.ylabel('frequency (Hz)')
        g.plot(d,t)

        g('unset multiplot')

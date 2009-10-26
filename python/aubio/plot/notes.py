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
	from numpy import array
	import Gnuplot, Gnuplot.funcutils
	d=[]
	x_widths = array(la[:,1]-la[:,0])/2.
	d.append(Gnuplot.Data(
	        la[:,0]+x_widths,               # x centers
	        la[:,2],                        # y centers
	        x_widths,                       # x errors
	        __notesheight*ones(len(la)),    # y errors
	        title=plot_title,with_=('boxxyerrorbars fs 3')))
	return d


def plotnote_withoutends(la,plot_title=None) :
        """ bug: fails drawing last note """
	from numpy import array
	import Gnuplot, Gnuplot.funcutils
        d=[]
        x_widths = array(la[1:,0]-la[:-1,0])/2;
        d.append(Gnuplot.Data(
                la[:-1,0]+x_widths,             # x centers
                la[:-1,1],                      # y centers
                x_widths,                       # x errors
                __notesheight*ones(len(la)-1),  # y errors
                title=plot_title,with_=('boxxyerrorbars fs 3')))
        return d

def plotnote_do(d,fileout=None):
    import Gnuplot, Gnuplot.funcutils
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


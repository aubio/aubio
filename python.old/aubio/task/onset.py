from aubio.task.task import task
from aubio.aubioclass import *

class taskonset(task):
	def __init__(self,input,output=None,params=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		task.__init__(self,input,params=params)
		self.opick = onsetpick(self.params.bufsize,
			self.params.hopsize,
			self.myvec,
			self.params.threshold,
			mode=self.params.onsetmode,
			dcthreshold=self.params.dcthreshold,
			derivate=self.params.derivate)
		self.olist = [] 
		self.ofunc = []
		self.maxofunc = 0
		self.last = 0
		if self.params.localmin:
			self.ovalist   = [0., 0., 0., 0., 0.]

	def __call__(self):
		task.__call__(self)
		isonset,val = self.opick.do(self.myvec)
		if (aubio_silence_detection(self.myvec(),self.params.silence)):
			isonset=0
		if self.params.storefunc:
			self.ofunc.append(val)
		if self.params.localmin:
			if val > 0: self.ovalist.append(val)
			else: self.ovalist.append(0)
			self.ovalist.pop(0)
		if (isonset > 0.):
			if self.params.localmin:
				# find local minima before peak 
				i=len(self.ovalist)-1
				while self.ovalist[i-1] < self.ovalist[i] and i > 0:
					i -= 1
				now = (self.frameread+1-i)
			else:
				now = self.frameread
			# take back delay
			if self.params.delay != 0.: now -= self.params.delay
			if now < 0 :
				now = 0
			if self.params.mintol:
				# prune doubled 
				if (now - self.last) > self.params.mintol:
					self.last = now
					return now, val
			else:
				return now, val 


	def fprint(self,foo):
		print self.params.step*foo[0]

	def eval(self,inputdata,ftru,mode='roc',vmode=''):
		from aubio.txtfile import read_datafile 
		from aubio.onsetcompare import onset_roc, onset_diffs, onset_rocloc
		ltru = read_datafile(ftru,depth=0)
		lres = []
		for i in range(len(inputdata)): lres.append(inputdata[i][0]*self.params.step)
		if vmode=='verbose':
			print "Running with mode %s" % self.params.onsetmode, 
			print " and threshold %f" % self.params.threshold, 
			print " on file", self.input
		#print ltru; print lres
		if mode == 'local':
			l = onset_diffs(ltru,lres,self.params.tol)
			mean = 0
			for i in l: mean += i
			if len(l): mean = "%.3f" % (mean/len(l))
			else: mean = "?0"
			return l, mean
		elif mode == 'roc':
			self.orig, self.missed, self.merged, \
				self.expc, self.bad, self.doubled = \
				onset_roc(ltru,lres,self.params.tol)
		elif mode == 'rocloc':
			self.v = {}
			self.v['orig'], self.v['missed'], self.v['Tm'], \
				self.v['expc'], self.v['bad'], self.v['Td'], \
				self.v['l'], self.v['labs'] = \
				onset_rocloc(ltru,lres,self.params.tol)

	def plot(self,onsets,ofunc,wplot,oplots,nplot=False):
		import Gnuplot, Gnuplot.funcutils
		import aubio.txtfile
		import os.path
		from numpy import arange, array, ones
		from aubio.onsetcompare import onset_roc

		x1,y1,y1p = [],[],[]
		oplot = []
		if self.params.onsetmode in ('mkl','kl'): ofunc[0:10] = [0] * 10

		self.lenofunc = len(ofunc) 
		self.maxofunc = max(ofunc)
		# onset detection function 
		downtime = arange(len(ofunc))*self.params.step
		oplot.append(Gnuplot.Data(downtime,ofunc,with_='lines',title=self.params.onsetmode))

		# detected onsets
		if not nplot:
			for i in onsets:
				x1.append(i[0]*self.params.step)
				y1.append(self.maxofunc)
				y1p.append(-self.maxofunc)
			#x1 = array(onsets)*self.params.step
			#y1 = self.maxofunc*ones(len(onsets))
			if x1:
				oplot.append(Gnuplot.Data(x1,y1,with_='impulses'))
				wplot.append(Gnuplot.Data(x1,y1p,with_='impulses'))

		oplots.append((oplot,self.params.onsetmode,self.maxofunc))

		# check if ground truth datafile exists
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		if not os.path.isfile(datafile):
			self.title = "" #"(no ground truth)"
		else:
			t_onsets = aubio.txtfile.read_datafile(datafile)
			x2 = array(t_onsets).resize(len(t_onsets))
			y2 = self.maxofunc*ones(len(t_onsets))
			wplot.append(Gnuplot.Data(x2,y2,with_='impulses'))
			
			tol = 0.050 

			orig, missed, merged, expc, bad, doubled = \
				onset_roc(x2,x1,tol)
			self.title = "GD %2.3f%% FP %2.3f%%" % \
				((100*float(orig-missed-merged)/(orig)),
				 (100*float(bad+doubled)/(orig)))


	def plotplot(self,wplot,oplots,outplot=None,extension=None,xsize=1.,ysize=1.,spectro=False):
		from aubio.gnuplot import gnuplot_create, audio_to_array, make_audio_plot, audio_to_spec
		import re
		# prepare the plot
		g = gnuplot_create(outplot=outplot, extension=extension)
		g('set title \'%s\'' % (re.sub('.*/','',self.input)))
		if spectro:
			g('set size %f,%f' % (xsize,1.3*ysize) )
		else:
			g('set size %f,%f' % (xsize,ysize) )
		g('set multiplot')

		# hack to align left axis
		g('set lmargin 3')
		g('set rmargin 6')

		if spectro:
			import Gnuplot
			minf = 50
			maxf = 500 
			data,time,freq = audio_to_spec(self.input,minf=minf,maxf=maxf)
			g('set size %f,%f' % (1.24*xsize , 0.34*ysize) )
			g('set origin %f,%f' % (-0.12,0.65*ysize))
			g('set xrange [0.:%f]' % time[-1]) 
			g('set yrange [%f:%f]' % (minf,maxf))
			g('set pm3d map')
			g('unset colorbox')
			g('set lmargin 0')
			g('set rmargin 0')
			g('set tmargin 0')
			g('set palette rgbformulae -25,-24,-32')
			g.xlabel('time (s)',offset=(0,1.))
			g.ylabel('freq (Hz)')
			g('set origin 0,%f' % (1.0*ysize) ) 
			g('set format x "%1.1f"')
			#if log:
			#	g('set yrange [%f:%f]' % (max(10,minf),maxf))
			#	g('set log y')
			g.splot(Gnuplot.GridData(data,time,freq, binary=1, title=''))
		else:
			# plot waveform and onsets
			time,data = audio_to_array(self.input)
			wplot = [make_audio_plot(time,data)] + wplot
			g('set origin 0,%f' % (0.7*ysize) )
			g('set size %f,%f' % (xsize,0.3*ysize))
			g('set format y "%1f"')
			g('set xrange [0:%f]' % max(time)) 
			g('set yrange [-1:1]') 
			g('set noytics')
			g('set y2tics -1,1')
			g.xlabel('time (s)',offset=(0,0.7))
			g.ylabel('amplitude')
			g.plot(*wplot)

		# default settings for next plots
		g('unset title')
		g('set format x ""')
		g('set format y "%3e"')
		g('set tmargin 0')
		g.xlabel('')

		N = len(oplots)
		y = 0.7*ysize # the vertical proportion of the plot taken by onset functions
		delta = 0.035 # the constant part of y taken by last plot label and data
		for i in range(N):
			# plot onset detection functions
			g('set size %f,%f' % ( xsize, (y-delta)/N))
			g('set origin 0,%f' % ((N-i-1)*(y-delta)/N + delta ))
			g('set nokey')
			g('set xrange [0:%f]' % (self.lenofunc*self.params.step))
			g('set yrange [0:%f]' % (1.1*oplots[i][2]))
			g('set y2tics ("0" 0, "%d" %d)' % (round(oplots[i][2]),round(oplots[i][2])))
			g.ylabel(oplots[i][1])
			if i == N-1:
				g('set size %f,%f' % ( xsize, (y-delta)/N + delta ) )
				g('set origin 0,0')
				g.xlabel('time (s)', offset=(0,0.7))
				g('set format x')
			g.plot(*oplots[i][0])

		g('unset multiplot')

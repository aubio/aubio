from aubio.task.task import task
from aubio.task.utils import * 
from aubio.aubioclass import *

class taskonset(task):
	def __init__(self,input,output=None,params=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		task.__init__(self,input,params=params)
		self.opick = onsetpick(self.params.bufsize,
			self.params.hopsize,
			self.channels,
			self.myvec,
			self.params.threshold,
			mode=get_onset_mode(self.params.onsetmode),
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
		if (isonset == 1):
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
		import numarray
		from aubio.onsetcompare import onset_roc

		x1,y1,y1p = [],[],[]
		oplot = []
		if self.params.onsetmode in ('mkl','kl'): ofunc[0:10] = [0] * 10

		self.lenofunc = len(ofunc) 
		self.maxofunc = max(ofunc)
		# onset detection function 
		downtime = numarray.arange(len(ofunc))*self.params.step
		oplot.append(Gnuplot.Data(downtime,ofunc,with='lines',title=self.params.onsetmode))

		# detected onsets
		if not nplot:
			for i in onsets:
				x1.append(i[0]*self.params.step)
				y1.append(self.maxofunc)
				y1p.append(-self.maxofunc)
			#x1 = numarray.array(onsets)*self.params.step
			#y1 = self.maxofunc*numarray.ones(len(onsets))
			if x1:
				oplot.append(Gnuplot.Data(x1,y1,with='impulses'))
				wplot.append(Gnuplot.Data(x1,y1p,with='impulses'))

		oplots.append(oplot)

		# check if ground truth datafile exists
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		if not os.path.isfile(datafile):
			self.title = "" #"(no ground truth)"
		else:
			t_onsets = aubio.txtfile.read_datafile(datafile)
			x2 = numarray.array(t_onsets).resize(len(t_onsets))
			y2 = self.maxofunc*numarray.ones(len(t_onsets))
			wplot.append(Gnuplot.Data(x2,y2,with='impulses'))
			
			tol = 0.050 

			orig, missed, merged, expc, bad, doubled = \
				onset_roc(x2,x1,tol)
			self.title = "GD %2.3f%% FP %2.3f%%" % \
				((100*float(orig-missed-merged)/(orig)),
				 (100*float(bad+doubled)/(orig)))


	def plotplot(self,wplot,oplots,outplot=None):
		from aubio.gnuplot import gnuplot_init, audio_to_array, make_audio_plot
		import re
		# audio data
		time,data = audio_to_array(self.input)
		wplot = [make_audio_plot(time,data)] + wplot
		self.title = self.input
		# prepare the plot
		g = gnuplot_init(outplot)

		g('set multiplot')

		# hack to align left axis
		g('set lmargin 6')
		g('set tmargin 0')
		g('set format x ""')
		g('set format y ""')
		g('set noytics')

		for i in range(len(oplots)):
			# plot onset detection functions
			g('set size 1,%f' % (0.7/(len(oplots))))
			g('set origin 0,%f' % (float(i)*0.7/(len(oplots))))
			g('set xrange [0:%f]' % (self.lenofunc*self.params.step))
			g.plot(*oplots[i])

		g('set tmargin 3.0')
		g('set xlabel "time (s)" 1,0')
		g('set format x "%1.1f"')

		g('set title \'%s %s\'' % (re.sub('.*/','',self.input),self.title))

		# plot waveform and onsets
		g('set size 1,0.3')
		g('set origin 0,0.7')
		g('set xrange [0:%f]' % max(time)) 
		g('set yrange [-1:1]') 
		g.ylabel('amplitude')
		g.plot(*wplot)
		
		g('unset multiplot')



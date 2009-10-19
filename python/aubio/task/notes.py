
from aubio.task import task
from aubio.aubioclass import *

class tasknotes(task):
	def __init__(self,input,output=None,params=None):
		task.__init__(self,input,params=params)
		self.opick = onsetpick(self.params.bufsize,
			self.params.hopsize,
			self.channels,
			self.myvec,
			self.params.threshold,
			mode=self.params.onsetmode,
			dcthreshold=self.params.dcthreshold,
			derivate=self.params.derivate)
		self.pitchdet  = pitch(mode=self.params.pitchmode,
			bufsize=self.params.pbufsize,
			hopsize=self.params.phopsize,
			channels=self.channels,
			samplerate=self.srate,
			omode=self.params.omode)
		self.olist = [] 
		self.ofunc = []
		self.maxofunc = 0
		self.last = -1000
		self.oldifreq = 0
		if self.params.localmin:
			self.ovalist   = [0., 0., 0., 0., 0.]

	def __call__(self):
		from aubio.median import short_find
		task.__call__(self)
		isonset,val = self.opick.do(self.myvec)
		if (aubio_silence_detection(self.myvec(),self.params.silence)):
			isonset=0
			freq = -1.
		else:
			freq = self.pitchdet(self.myvec)
		minpitch = self.params.pitchmin
		maxpitch = self.params.pitchmax
		if maxpitch and freq > maxpitch : 
			freq = -1.
		elif minpitch and freq < minpitch :
			freq = -1.
		freq = aubio_freqtomidi(freq)
		if self.params.pitchsmooth:
			self.shortlist.append(freq)
			self.shortlist.pop(0)
			smoothfreq = short_find(self.shortlist,
				len(self.shortlist)/2)
			freq = smoothfreq
		now = self.frameread
		ifreq = int(round(freq))
		if self.oldifreq == ifreq:
			self.oldifreq = ifreq
		else:
			self.oldifreq = ifreq
			ifreq = 0 
		# take back delay
		if self.params.delay != 0.: now -= self.params.delay
		if now < 0 :
			now = 0
		if (isonset == 1):
			if self.params.mintol:
				# prune doubled 
				if (now - self.last) > self.params.mintol:
					self.last = now
					return now, 1, freq, ifreq
				else:
					return now, 0, freq, ifreq
			else:
				return now, 1, freq, ifreq 
		else:
			return now, 0, freq, ifreq


	def fprint(self,foo):
		print self.params.step*foo[0], foo[1], foo[2], foo[3]

	def compute_all(self):
		""" Compute data """
    		now, onset, freq, ifreq = [], [], [], []
		while(self.readsize==self.params.hopsize):
			n, o, f, i = self()
			now.append(n*self.params.step)
			onset.append(o)
			freq.append(f)
			ifreq.append(i)
			if self.params.verbose:
				self.fprint((n,o,f,i))
    		return now, onset, freq, ifreq 

	def plot(self,now,onset,freq,ifreq,oplots):
		import Gnuplot

		oplots.append(Gnuplot.Data(now,freq,with_='lines',
			title=self.params.pitchmode))
		oplots.append(Gnuplot.Data(now,ifreq,with_='lines',
			title=self.params.pitchmode))

		temponsets = []
		for i in onset:
			temponsets.append(i*1000)
		oplots.append(Gnuplot.Data(now,temponsets,with_='impulses',
			title=self.params.pitchmode))

	def plotplot(self,wplot,oplots,outplot=None,multiplot = 0):
		from aubio.gnuplot import gnuplot_init, audio_to_array, make_audio_plot
		import re
		import Gnuplot
		# audio data
		time,data = audio_to_array(self.input)
		f = make_audio_plot(time,data)

		# check if ground truth exists
		#timet,pitcht = self.gettruth()
		#if timet and pitcht:
		#	oplots = [Gnuplot.Data(timet,pitcht,with_='lines',
		#		title='ground truth')] + oplots

		t = Gnuplot.Data(0,0,with_='impulses') 

		g = gnuplot_init(outplot)
		g('set title \'%s\'' % (re.sub('.*/','',self.input)))
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
		g('set yrange [20:100]')
		g('set key right top')
		g('set noclip one') 
		#g('set format x ""')
		#g('set log y')
		#g.xlabel('time (s)')
		g.ylabel('f0 (Hz)')
		if multiplot:
			for i in range(len(oplots)):
				# plot onset detection functions
				g('set size 1,%f' % (0.7/(len(oplots))))
				g('set origin 0,%f' % (float(i)*0.7/(len(oplots))))
				g('set xrange [0:%f]' % max(time))
				g.plot(oplots[i])
		else:
			g.plot(*oplots)
		#g('unset multiplot')


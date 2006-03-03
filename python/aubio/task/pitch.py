from aubio.task.task import task
from aubio.task.silence import tasksilence
from aubio.task.utils import * 
from aubio.aubioclass import *

class taskpitch(task):
	def __init__(self,input,params=None):
		task.__init__(self,input,params=params)
		self.shortlist = [0. for i in range(self.params.pitchsmooth)]
		self.pitchdet  = pitchdetection(mode=get_pitch_mode(self.params.pitchmode),
			bufsize=self.params.bufsize,
			hopsize=self.params.hopsize,
			channels=self.channels,
			samplerate=self.srate,
			omode=self.params.omode)

	def __call__(self):
		from aubio.median import short_find
		task.__call__(self)
		if (aubio_silence_detection(self.myvec(),self.params.silence)==1):
			freq = -1.
		else:
			freq = self.pitchdet(self.myvec)
		minpitch = self.params.pitchmin
		maxpitch = self.params.pitchmax
		if maxpitch and freq > maxpitch : 
			freq = -1.
		elif minpitch and freq < minpitch :
			freq = -1.
		if self.params.pitchsmooth:
			self.shortlist.append(freq)
			self.shortlist.pop(0)
			smoothfreq = short_find(self.shortlist,
				len(self.shortlist)/2)
			return smoothfreq
		else:
			return freq

	def compute_all(self):
		""" Compute data """
    		mylist    = []
		while(self.readsize==self.params.hopsize):
			freq = self()
			mylist.append(freq)
			if self.params.verbose:
				self.fprint("%s\t%s" % (self.frameread*self.params.step,freq))
    		return mylist

	def gettruth(self):
		""" extract ground truth array in frequency """
		import os.path
		""" from wavfile.txt """
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		""" from file.<midinote>.wav """
		# FIXME very weak check
		floatpit = self.input.split('.')[-2]
		if not os.path.isfile(datafile) and len(self.input.split('.')) < 3:
			print "no ground truth "
			return False,False
		elif floatpit:
			try:
				self.truth = aubio_miditofreq(float(floatpit))
				print "ground truth found in filename:", self.truth
				tasksil = tasksilence(self.input)
				time,pitch =[],[]
				while(tasksil.readsize==tasksil.params.hopsize):
					tasksil()
					time.append(tasksil.params.step*tasksil.frameread)
					if not tasksil.issilence:
						pitch.append(self.truth)
					else:
						pitch.append(-1.)
				return time,pitch #0,aubio_miditofreq(float(floatpit))
			except ValueError:
				# FIXME very weak check
				if not os.path.isfile(datafile):
					print "no ground truth found"
					return 0,0
				else:
					from aubio.txtfile import read_datafile
					values = read_datafile(datafile)
					time, pitch = [], []
					for i in range(len(values)):
						time.append(values[i][0])
						pitch.append(values[i][1])
					return time,pitch

	def eval(self,results):
		def mmean(l):
			return sum(l)/max(float(len(l)),1)

		from aubio.median import percental 
		timet,pitcht = self.gettruth()
		res = []
		for i in results:
			#print i,self.truth
			if i <= 0: pass
			else: res.append(self.truth-i)
		if not res or len(res) < 3: 
			avg = self.truth; med = self.truth 
		else:
			avg = mmean(res) 
			med = percental(res,len(res)/2) 
		return self.truth, self.truth-med, self.truth-avg

	def neweval(self,results):
		timet,pitcht = self.gettruth()
		for i in timet:
			print results[i]
		return self.truth, self.truth-med, self.truth-avg

	def plot(self,pitch,wplot,oplots,outplot=None):
		import numarray
		import Gnuplot

		self.eval(pitch)
		downtime = self.params.step*numarray.arange(len(pitch))
		oplots.append(Gnuplot.Data(downtime,pitch,with='lines',
			title=self.params.pitchmode))

			
	def plotplot(self,wplot,oplots,outplot=None,multiplot = 1):
		from aubio.gnuplot import gnuplot_init, audio_to_array, make_audio_plot
		import re
		import Gnuplot
		# audio data
		time,data = audio_to_array(self.input)
		f = make_audio_plot(time,data)

		# check if ground truth exists
		timet,pitcht = self.gettruth()
		if timet and pitcht:
			oplots = [Gnuplot.Data(timet,pitcht,with='lines',
				title='ground truth')] + oplots

		t = Gnuplot.Data(0,0,with='impulses') 

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
		g('set yrange [100:%f]' % self.params.pitchmax) 
		g('set key right top')
		g('set noclip one') 
		g('set format x ""')
		g('set log y')
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
		g('unset multiplot')


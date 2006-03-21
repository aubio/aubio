from aubio.task.task import task
from aubio.task.silence import tasksilence
from aubio.task.utils import * 
from aubio.aubioclass import *

class taskpitch(task):
	def __init__(self,input,params=None):
		task.__init__(self,input,params=params)
		self.shortlist = [0. for i in range(self.params.pitchsmooth)]
		if self.params.pitchmode == 'yinfft':
			yinthresh = self.params.yinfftthresh
		elif self.params.pitchmode == 'yin':
			yinthresh = self.params.yinthresh
		else:
			yinthresh = 0.
		self.pitchdet  = pitchdetection(mode=get_pitch_mode(self.params.pitchmode),
			bufsize=self.params.bufsize,
			hopsize=self.params.hopsize,
			channels=self.channels,
			samplerate=self.srate,
			omode=self.params.omode,
			yinthresh=yinthresh)

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
				self.truth = float(floatpit)
				#print "ground truth found in filename:", self.truth
				tasksil = tasksilence(self.input,params=self.params)
				time,pitch =[],[]
				while(tasksil.readsize==tasksil.params.hopsize):
					tasksil()
					time.append(tasksil.params.step*(tasksil.frameread))
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
						if values[i][1] == 0.0:
							pitch.append(-1.)
						else:
							pitch.append(aubio_freqtomidi(values[i][1]))
					return time,pitch

	def oldeval(self,results):
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

	def eval(self,pitch,tol=0.5):
		timet,pitcht = self.gettruth()
		pitch = [aubio_freqtomidi(i) for i in pitch]
		for i in range(len(pitch)):
			if pitch[i] == "nan" or pitch[i] == -1:
				pitch[i] = -1
		time = [ i*self.params.step for i in range(len(pitch)) ]
		#print len(timet),len(pitcht)
		#print len(time),len(pitch)
		if len(timet) != len(time):
			time = time[1:len(timet)+1]
			pitch = pitch[1:len(pitcht)+1]
			#pitcht = [aubio_freqtomidi(i) for i in pitcht]
			for i in range(len(pitcht)):
				if pitcht[i] == "nan" or pitcht[i] == "-inf" or pitcht[i] == -1:
					pitcht[i] = -1
		assert len(timet) == len(time)
		assert len(pitcht) == len(pitch)
		osil, esil, opit, epit, echr = 0, 0, 0, 0, 0
		for i in range(len(pitcht)):
			if pitcht[i] == -1: # currently silent
				osil += 1 # count a silence
				if pitch[i] <= 0. or pitch[i] == "nan": 
					esil += 1 # found a silence
			else:
				opit +=1
				if abs(pitcht[i] - pitch[i]) < tol:
					epit += 1
					echr += 1
				elif abs(pitcht[i] - pitch[i]) % 12. < tol:
					echr += 1
				#else:
				#	print timet[i], pitcht[i], time[i], pitch[i]
		#print "origsilence", "foundsilence", "origpitch", "foundpitch", "orig pitchroma", "found pitchchroma"
		#print 100.*esil/float(osil), 100.*epit/float(opit), 100.*echr/float(opit)
		return osil, esil, opit, epit, echr

	def plot(self,pitch,wplot,oplots,outplot=None):
		import numarray
		import Gnuplot

		downtime = self.params.step*numarray.arange(len(pitch))
		pitch = [aubio_freqtomidi(i) for i in pitch]
		oplots.append(Gnuplot.Data(downtime,pitch,with='lines',
			title=self.params.pitchmode))

			
	def plotplot(self,wplot,oplots,outplot=None,multiplot = 0, midi = 1):
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
		if not midi:
			g('set log y')
			#g.xlabel('time (s)')
			g.ylabel('f0 (Hz)')
			g('set yrange [100:%f]' % self.params.pitchmax) 
		else: 
			g.ylabel('pitch (midi)')
			g('set yrange [%f:%f]' % (aubio_freqtomidi(self.params.pitchmin), aubio_freqtomidi(self.params.pitchmax)))
		g('set key right top')
		g('set noclip one') 
		g('set format x ""')
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


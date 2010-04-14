from aubio.task.task import task
from aubio.task.silence import tasksilence
from aubio.aubioclass import *

class taskpitch(task):
	def __init__(self,input,params=None):
		task.__init__(self,input,params=params)
		self.shortlist = [0. for i in range(self.params.pitchsmooth)]
		if self.params.pitchmode == 'yin':
			tolerance = self.params.yinthresh
		elif self.params.pitchmode == 'yinfft':
			tolerance = self.params.yinfftthresh
		else:
			tolerance = 0.
		self.pitchdet	= pitch(mode=self.params.pitchmode,
			bufsize=self.params.bufsize,
			hopsize=self.params.hopsize,
			samplerate=self.srate,
			omode=self.params.omode,
			tolerance = tolerance)

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
				return time,pitch
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
		time = [ (i+self.params.pitchdelay)*self.params.step for i in range(len(pitch)) ]
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

	def plot(self,pitch,wplot,oplots,titles,outplot=None):
		import Gnuplot

		time = [ (i+self.params.pitchdelay)*self.params.step for i in range(len(pitch)) ]
		pitch = [aubio_freqtomidi(i) for i in pitch]
		oplots.append(Gnuplot.Data(time,pitch,with_='lines',
			title=self.params.pitchmode))
		titles.append(self.params.pitchmode)

			
	def plotplot(self,wplot,oplots,titles,outplot=None,extension=None,xsize=1.,ysize=1.,multiplot = 1, midi = 1, truth = 1):
		from aubio.gnuplot import gnuplot_create , audio_to_array, make_audio_plot
		import re
		import Gnuplot

		# check if ground truth exists
		if truth:
			timet,pitcht = self.gettruth()
			if timet and pitcht:
				oplots = [Gnuplot.Data(timet,pitcht,with_='lines',
					title='ground truth')] + oplots

		g = gnuplot_create(outplot=outplot, extension=extension)
		g('set title \'%s\'' % (re.sub('.*/','',self.input)))
		g('set size %f,%f' % (xsize,ysize) )
		g('set multiplot')
		# hack to align left axis
		g('set lmargin 4')
		g('set rmargin 4')
    # plot waveform
		time,data = audio_to_array(self.input)
		wplot = [make_audio_plot(time,data)]
		g('set origin 0,%f' % (0.7*ysize) )
		g('set size %f,%f' % (xsize,0.3*ysize))
		#g('set format y "%1f"')
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
		g('set noclip one') 

		if not midi:
			g('set log y')
			#g.xlabel('time (s)')
			g.ylabel('f0 (Hz)')
			g('set yrange [100:%f]' % self.params.pitchmax) 
		else: 
			g.ylabel('midi')
			g('set yrange [%f:%f]' % (aubio_freqtomidi(self.params.pitchmin), aubio_freqtomidi(self.params.pitchmax)))
			g('set y2tics %f,%f' % (round(aubio_freqtomidi(self.params.pitchmin)+.5),12))
		
		if multiplot:
			N = len(oplots)
			y = 0.7*ysize # the vertical proportion of the plot taken by onset functions
			delta = 0.035 # the constant part of y taken by last plot label and data
			for i in range(N):
				# plot pitch detection functions
				g('set size %f,%f' % ( xsize, (y-delta)/N))
				g('set origin 0,%f' % ((N-i-1)*(y-delta)/N + delta ))
				g('set nokey')
				g('set xrange [0:%f]' % max(time))
				g.ylabel(titles[i])
				if i == N-1:
					g('set size %f,%f' % (xsize, (y-delta)/N + delta ) )
					g('set origin 0,0')
					g.xlabel('time (s)', offset=(0,0.7))
					g('set format x')
				g.plot(oplots[i])
		else:
			g('set key right top')
			g.plot(*oplots)
		g('unset multiplot')


from aubioclass import * 

def get_onset_mode(nvalue):
        """ utility function to convert a string to aubio_onsetdetection_type """
	if   nvalue == 'complexdomain' or nvalue == 'complex' :
		 return aubio_onset_complex
	elif nvalue == 'hfc'           :
		 return aubio_onset_hfc
	elif nvalue == 'phase'         :
		 return aubio_onset_phase
	elif nvalue == 'specdiff'      :
		 return aubio_onset_specdiff
	elif nvalue == 'energy'        :
		 return aubio_onset_energy
	elif nvalue == 'kl'            :
		 return aubio_onset_kl
	elif nvalue == 'mkl'           :
		 return aubio_onset_mkl
	elif nvalue == 'dual'          :
		 return 'dual'
	else:
		 import sys
		 print "unknown onset detection function selected"
		 sys.exit(1)

def get_pitch_mode(nvalue):
        """ utility function to convert a string to aubio_pitchdetection_type """
	if   nvalue == 'mcomb'  :
		 return aubio_pitch_mcomb
	elif nvalue == 'yin'    :
		 return aubio_pitch_yin
	elif nvalue == 'fcomb'  :
		 return aubio_pitch_fcomb
	elif nvalue == 'schmitt':
		 return aubio_pitch_schmitt
	else:
		 import sys
		 print "error: unknown pitch detection function selected"
		 sys.exit(1)

def check_onset_mode(option, opt, value, parser):
        """ wrapper function to convert a list of modes to 
		aubio_onsetdetection_type """
        nvalues = parser.rargs[0].split(',')
        val =  []
        for nvalue in nvalues:
		val.append(get_onset_mode(nvalue))
                setattr(parser.values, option.dest, val)

def check_pitch_mode(option, opt, value, parser):
        """ utility function to convert a string to aubio_pitchdetection_type"""
        nvalues = parser.rargs[0].split(',')
        val = []
        for nvalue in nvalues:
		val.append(get_pitch_mode(nvalue))
                setattr(parser.values, option.dest, val)

def check_pitchm_mode(option, opt, value, parser):
        """ utility function to convert a string to aubio_pitchdetection_mode """
        nvalue = parser.rargs[0]
        if   nvalue == 'freq'  :
                 setattr(parser.values, option.dest, aubio_pitchm_freq)
        elif nvalue == 'midi'  :
                 setattr(parser.values, option.dest, aubio_pitchm_midi)
        elif nvalue == 'cent'  :
                 setattr(parser.values, option.dest, aubio_pitchm_cent)
        elif nvalue == 'bin'   :
                 setattr(parser.values, option.dest, aubio_pitchm_bin)
        else:
                 import sys
                 print "error: unknown pitch detection output selected"
                 sys.exit(1)

class taskparams(object):
	""" default parameters for task classes """
	def __init__(self,input=None,output=None):
		self.silence = -70
		self.derivate = False
		self.localmin = False
		self.delay = 4.
		self.storefunc = False
		self.bufsize = 512
		self.hopsize = 256
		self.samplerate = 44100
		self.tol = 0.05
		self.mintol = 0.0
		self.step = float(self.hopsize)/float(self.samplerate)
		self.threshold = 0.1
		self.onsetmode = 'dual'
		self.pitchmode = 'yin'
		self.pitchsmooth = 20
		self.pitchmin=100.
		self.pitchmax=1500.
		self.dcthreshold = -1.
		self.omode = aubio_pitchm_freq
		self.verbose   = False

class task(taskparams):
	""" default template class to apply tasks on a stream """
	def __init__(self,input,output=None,params=None):
		""" open the input file and initialize default argument 
		parameters should be set *before* calling this method.
		"""
		import time
		self.tic = time.time()
		if params == None: self.params = taskparams()
		else: self.params = params
		self.frameread = 0
		self.readsize  = self.params.hopsize
		self.input     = input
		self.filei     = sndfile(self.input)
		self.srate     = self.filei.samplerate()
		self.channels  = self.filei.channels()
		self.params.step = float(self.params.hopsize)/float(self.srate)
		self.myvec     = fvec(self.params.hopsize,self.channels)
		self.output    = output

	def __call__(self):
		self.readsize = self.filei.read(self.params.hopsize,self.myvec)
		self.frameread += 1
		
	def compute_all(self):
		""" Compute data """
    		mylist    = []
		while(self.readsize==self.params.hopsize):
			tmp = self()
			if tmp: 
				mylist.append(tmp)
				if self.params.verbose:
					self.fprint(tmp)
    		return mylist
	
	def fprint(self,foo):
		print foo

	def eval(self,results):
		""" Eval data """
		pass

	def plot(self):
		""" Plot data """
		pass

	def time(self):
		import time
		print "CPU time is now %f seconds," % time.clock(),
		print "task execution took %f seconds" % (time.time() - self.tic)

class tasksilence(task):
	wassilence = 1
	issilence  = 1
	def __call__(self):
		task.__call__(self)
		if (aubio_silence_detection(self.myvec(),self.params.silence)==1):
			if self.wassilence == 1: self.issilence = 1
			else: self.issilence = 2
			self.wassilence = 1
		else: 
			if self.wassilence <= 0: self.issilence = 0
			else: self.issilence = -1 
			self.wassilence = 0
		if self.issilence == -1:
			return max(self.frameread-self.params.delay,0.), -1
		elif self.issilence == 2:
			return max(self.frameread+self.params.delay,0.), 2 

	def fprint(self,foo):
		print self.params.step*foo[0],
		if foo[1] == 2: print "OFF"
		else: print "ON"

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
		from median import short_find
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
		""" big hack to extract midi note from /path/to/file.<midinote>.wav """
		floatpit = self.input.split('.')[-2]
		try:
			return aubio_miditofreq(float(floatpit))
		except ValueError:
			print "ERR: no truth file found"
			return 0

	def eval(self,results):
		def mmean(l):
			return sum(l)/max(float(len(l)),1)

		from median import percental 
		self.truth = self.gettruth()
		res = []
		for i in results:
			if i <= 0: pass
			else: res.append(self.truth-i)
		if not res: 
			avg = self.truth; med = self.truth 
		else:
			avg = mmean(res) 
			med = percental(res,len(res)/2) 
		return self.truth, self.truth-med, self.truth-avg

	def plot(self,pitch,wplot,oplots,outplot=None):
		from aubio.txtfile import read_datafile
		import os.path
		import numarray
		import Gnuplot

		downtime = self.params.step*numarray.arange(len(pitch))
		oplots.append(Gnuplot.Data(downtime,pitch,with='lines',
			title=self.params.pitchmode))

		# check if ground truth exists
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		if not os.path.isfile(datafile):
			self.title = "" #"truth file not found"
			t = Gnuplot.Data(0,0,with='impulses') 
		else:
			self.title = "" #"truth file plotting not implemented yet"
			values = read_datafile(datafile)
			if (len(datafile[0])) > 1:
				time, pitch = [], []
				for i in range(len(values)):
					time.append(values[i][0])
					pitch.append(values[i][1])
				oplots.append(Gnuplot.Data(time,pitch,with='lines',
					title='ground truth'))
			
	def plotplot(self,wplot,oplots,outplot=None):
		from aubio.gnuplot import gnuplot_init, audio_to_array, make_audio_plot
		import re
		# audio data
		time,data = audio_to_array(self.input)
		f = make_audio_plot(time,data)

		g = gnuplot_init(outplot)
		g('set title \'%s %s\'' % (re.sub('.*/','',self.input),self.title))
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
		g('set yrange [40:%f]' % self.params.pitchmax) 
		g('set key right top')
		g('set noclip one') 
		g('set format x ""')
		#g.xlabel('time (s)')
		g.ylabel('frequency (Hz)')
		multiplot = 1
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
		from txtfile import read_datafile 
		from onsetcompare import onset_roc, onset_diffs, onset_rocloc
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

		# check if datafile exists truth
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		if not os.path.isfile(datafile):
			self.title = "" #"(no ground truth)"
			t = Gnuplot.Data(0,0,with='impulses') 
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

class taskcut(task):
	def __init__(self,input,slicetimes,params=None,output=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		task.__init__(self,input,output=None,params=params)
		self.newname   = "%s%s%09.5f%s%s" % (self.input.split(".")[0].split("/")[-1],".",
					self.frameread*self.params.step,".",self.input.split(".")[-1])
		self.fileo	= sndfile(self.newname,model=self.filei)
		self.myvec	= fvec(self.params.hopsize,self.channels)
		self.mycopy	= fvec(self.params.hopsize,self.channels)
		self.slicetimes = slicetimes 

	def __call__(self):
		task.__call__(self)
		# write to current file
		if len(self.slicetimes) and self.frameread >= self.slicetimes[0][0]:
			self.slicetimes.pop(0)
			# write up to 1st zero crossing
			zerocross = 0
			while ( abs( self.myvec.get(zerocross,0) ) > self.params.zerothres ):
				zerocross += 1
			writesize = self.fileo.write(zerocross,self.myvec)
			fromcross = 0
			while (zerocross < self.readsize):
				for i in range(self.channels):
					self.mycopy.set(self.myvec.get(zerocross,i),fromcross,i)
					fromcross += 1
					zerocross += 1
			del self.fileo
			self.fileo = sndfile("%s%s%09.5f%s%s" % 
				(self.input.split(".")[0].split("/")[-1],".",
				self.frameread*self.params.step,".",self.input.split(".")[-1]),model=self.filei)
			writesize = self.fileo.write(fromcross,self.mycopy)
		else:
			writesize = self.fileo.write(self.readsize,self.myvec)

class taskbeat(taskonset):
	def __init__(self,input,params=None,output=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		taskonset.__init__(self,input,output=None,params=params)
		self.btwinlen  = 512**2/self.params.hopsize
		self.btstep    = self.btwinlen/4
		self.btoutput  = fvec(self.btstep,self.channels)
		self.dfframe   = fvec(self.btwinlen,self.channels)
		self.bt	       = beattracking(self.btwinlen,self.channels)
		self.pos2      = 0

	def __call__(self):
		taskonset.__call__(self)
		# write to current file
                if self.pos2 == self.btstep - 1 : 
                        self.bt.do(self.dfframe,self.btoutput)
                        for i in range (self.btwinlen - self.btstep):
                                self.dfframe.set(self.dfframe.get(i+self.btstep,0),i,0) 
                        for i in range(self.btwinlen - self.btstep, self.btwinlen): 
                                self.dfframe.set(0,i,0)
                        self.pos2 = -1;
                self.pos2 += 1
		val = self.opick.pp.getval()
		self.dfframe.set(val,self.btwinlen - self.btstep + self.pos2,0)
                i=0
                for i in range(1,int( self.btoutput.get(0,0) ) ):
                        if self.pos2 == self.btoutput.get(i,0) and \
				aubio_silence_detection(self.myvec(),
					self.params.silence)!=1: 
				return self.frameread, 0 
	
	def eval(self,results):
		pass

from aubioclass import * 
from bench.node import bench

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


#def getonsets(filein,threshold=0.2,silence=-70.,bufsize=1024,hopsize=512,
#                mode='dual',localmin=False,storefunc=False,derivate=False):
#        frameread = 0
#        filei     = sndfile(filein)
#        channels  = filei.channels()
#        myvec     = fvec(hopsize,channels)
#        readsize  = filei.read(hopsize,myvec)
#        opick     = onsetpick(bufsize,hopsize,channels,myvec,threshold,
#                         mode=mode,derivate=derivate)
#        mylist    = list()
#        if localmin:
#                ovalist   = [0., 0., 0., 0., 0.]
#        ofunclist = []
#        while(readsize):
#                readsize = filei.read(hopsize,myvec)
#                isonset,val = opick.do(myvec)
#                if (aubio_silence_detection(myvec(),silence)):
#                        isonset=0
#                if localmin:
#                        if val > 0: ovalist.append(val)
#                        else: ovalist.append(0)
#                        ovalist.pop(0)
#                if storefunc:
#                        ofunclist.append(val)
#                if (isonset == 1):
#                        if localmin:
#                                i=len(ovalist)-1
#                                # find local minima before peak 
#                                while ovalist[i-1] < ovalist[i] and i > 0:
#                                        i -= 1
#                                now = (frameread+1-i)
#                        else:
#                                now = frameread
#                        if now > 0 :
#                                mylist.append(now)
#                        else:
#                                now = 0
#                                mylist.append(now)
#                frameread += 1
#        return mylist, ofunclist
#
#def cutfile(filein,slicetimes,zerothres=0.008,bufsize=1024,hopsize=512):
#    frameread = 0
#    readsize  = hopsize 
#    filei     = sndfile(filein)
#    framestep = hopsize/(filei.samplerate()+0.)
#    channels  = filei.channels()
#    newname   = "%s%s%09.5f%s%s" % (filein.split(".")[0].split("/")[-1],".",
#                frameread*framestep,".",filein.split(".")[-1])
#    fileo     = sndfile(newname,model=filei)
#    myvec     = fvec(hopsize,channels)
#    mycopy    = fvec(hopsize,channels)
#    while(readsize==hopsize):
#        readsize = filei.read(hopsize,myvec)
#        # write to current file
#        if len(slicetimes) and frameread >= slicetimes[0]:
#            slicetimes.pop(0)
#            # write up to 1st zero crossing
#            zerocross = 0
#            while ( abs( myvec.get(zerocross,0) ) > zerothres ):
#                zerocross += 1
#            writesize = fileo.write(zerocross,myvec)
#            fromcross = 0
#            while (zerocross < readsize):
#                for i in range(channels):
#                    mycopy.set(myvec.get(zerocross,i),fromcross,i)
#                    fromcross += 1
#                    zerocross += 1
#            del fileo
#            fileo = sndfile("%s%s%09.5f%s%s" % 
#                (filein.split(".")[0].split("/")[-1],".",
#                frameread*framestep,".",filein.split(".")[-1]),model=filei)
#            writesize = fileo.write(fromcross,mycopy)
#        else:
#            writesize = fileo.write(readsize,myvec)
#        frameread += 1
#    del fileo
#
#
#def getsilences(filein,hopsize=512,silence=-70):
#    frameread = 0
#    filei     = sndfile(filein)
#    srate     = filei.samplerate()
#    channels  = filei.channels()
#    myvec     = fvec(hopsize,channels)
#    readsize  = filei.read(hopsize,myvec)
#    mylist    = []
#    wassilence = 0
#    while(readsize==hopsize):
#        readsize = filei.read(hopsize,myvec)
#        if (aubio_silence_detection(myvec(),silence)==1):
#            if wassilence == 0:
#                mylist.append(frameread)
#                wassilence == 1
#        else: wassilence = 0
#        frameread += 1
#    return mylist
#
#
#def getpitch(filein,mode=aubio_pitch_mcomb,bufsize=1024,hopsize=512,omode=aubio_pitchm_freq,
#        samplerate=44100.,silence=-70):
#    frameread = 0
#    filei     = sndfile(filein)
#    srate     = filei.samplerate()
#    channels  = filei.channels()
#    myvec     = fvec(hopsize,channels)
#    readsize  = filei.read(hopsize,myvec)
#    pitchdet  = pitchdetection(mode=mode,bufsize=bufsize,hopsize=hopsize,
#                         channels=channels,samplerate=srate,omode=omode)
#    mylist    = []
#    while(readsize==hopsize):
#        readsize = filei.read(hopsize,myvec)
#        freq = pitchdet(myvec)
#        #print "%.3f     %.2f" % (now,freq)
#        if (aubio_silence_detection(myvec(),silence)!=1):
#                mylist.append(freq)
#        else: 
#                mylist.append(-1.)
#        frameread += 1
#    return mylist


class taskparams(object):
	""" default parameters for task classes """
	def __init__(self,input=None,output=None):
		self.silence = -70
		self.derivate = False
		self.localmin = False
		self.storefunc = False
		self.bufsize = 512
		self.hopsize = 256
		self.samplerate = 44100
		self.tol = 0.05
		self.step = float(self.hopsize)/float(self.samplerate)
		self.threshold = 0.1
		self.onsetmode = 'dual'
		self.pitchmode = 'yin'
		self.omode = aubio_pitchm_freq

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
		self.step      = float(self.srate)/float(self.params.hopsize)
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
			if tmp: mylist.append(tmp)
    		return mylist

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
			return -1, self.frameread 
		elif self.issilence == 2:
			return 2, self.frameread 

class taskpitch(task):
	def __init__(self,input,params=None):
		task.__init__(self,input,params=params)
		self.pitchdet  = pitchdetection(mode=get_pitch_mode(self.params.pitchmode),
			bufsize=self.params.bufsize,
			hopsize=self.params.hopsize,
			channels=self.channels,
			samplerate=self.srate,
			omode=self.params.omode)

	def __call__(self):
		#print "%.3f     %.2f" % (now,freq)
		task.__call__(self)
		freq = self.pitchdet(self.myvec)
		if (aubio_silence_detection(self.myvec(),self.params.silence)!=1):
			return freq
		else: 
			return -1.

	def gettruth(self):
		""" big hack to extract midi note from /path/to/file.<midinote>.wav """
		floatpit = self.input.split('.')[-2]
		try:
			return float(floatpit)
		except ValueError:
			print "ERR: no truth file found"
			return 0

	def eval(self,results):
		from median import short_find 
		self.truth = self.gettruth()
		num = 0
		sum = 0
		res = []
		for i in results:
			if i == -1: pass
			else: 
				res.append(i)
				sum += i
				num += 1
		if num == 0: 
			avg = 0; med = 0
		else:
			avg = aubio_freqtomidi(sum / float(num))
			med = aubio_freqtomidi(short_find(res,len(res)/2))
		avgdist = self.truth - avg
		meddist = self.truth - med
		return avgdist, meddist

	def plot(self,pitch,outplot=None):
		from aubio.gnuplot import plot_pitch
		plot_pitch(self.input, 
			pitch, 
			samplerate=float(self.srate), 
			hopsize=self.params.hopsize, 
			outplot=outplot)


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
			derivate=self.params.derivate)
		self.olist = [] 
		self.ofunc = []
		self.d,self.d2 = [],[]
		self.maxofunc = 0
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
                                i=len(self.ovalist)-1
                                # find local minima before peak 
                                while self.ovalist[i-1] < self.ovalist[i] and i > 0:
                                        i -= 1
                                now = (self.frameread+1-i)
                        else:
                                now = self.frameread
                        if now < 0 :
                                now = 0
			return now, val 


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

	def plot(self,onsets,ofunc):
		import Gnuplot, Gnuplot.funcutils
		import aubio.txtfile
		import os.path
		import numarray
		from aubio.onsetcompare import onset_roc

		self.lenofunc = len(ofunc) 
		self.maxofunc = max(max(ofunc), self.maxofunc)
		# onset detection function 
		downtime = numarray.arange(len(ofunc))/self.step
		self.d.append(Gnuplot.Data(downtime,ofunc,with='lines'))

		# detected onsets
		x1 = numarray.array(onsets)/self.step
		y1 = self.maxofunc*numarray.ones(len(onsets))
		self.d.append(Gnuplot.Data(x1,y1,with='impulses'))
		self.d2.append(Gnuplot.Data(x1,-y1,with='impulses'))

		# check if datafile exists truth
		datafile = self.input.replace('.wav','.txt')
		if datafile == self.input: datafile = ""
		if not os.path.isfile(datafile):
			self.title = "truth file not found"
			t = Gnuplot.Data(0,0,with='impulses') 
		else:
			t_onsets = aubio.txtfile.read_datafile(datafile)
			y2 = self.maxofunc*numarray.ones(len(t_onsets))
			x2 = numarray.array(t_onsets).resize(len(t_onsets))
			self.d2.append(Gnuplot.Data(x2,y2,with='impulses'))
			
			tol = 0.050 

			orig, missed, merged, expc, bad, doubled = \
				onset_roc(x2,x1,tol)
			self.title = "GD %2.3f%% FP %2.3f%%" % \
				((100*float(orig-missed-merged)/(orig)),
				 (100*float(bad+doubled)/(orig)))


	def plotplot(self,outplot=None):
		from aubio.gnuplot import gnuplot_init, audio_to_array, make_audio_plot
		import re
		# audio data
		time,data = audio_to_array(self.input)
		self.d2.append(make_audio_plot(time,data))
		# prepare the plot
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
		g.plot(*self.d2)
		
		g('unset title')

		# plot onset detection function
		g('set size 1,0.7')
		g('set origin 0,0')
		g('set xrange [0:%f]' % (self.lenofunc/self.step))
		g('set yrange [0:%f]' % (self.maxofunc*1.01))
		g.xlabel('time')
		g.ylabel('onset detection value')
		g.plot(*self.d)

		g('unset multiplot')

class taskcut(task):
	def __init__(self,input,slicetimes,params=None,output=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		task.__init__(self,input,output=None,params=params)
		self.newname   = "%s%s%09.5f%s%s" % (self.input.split(".")[0].split("/")[-1],".",
					self.frameread/self.step,".",self.input.split(".")[-1])
		self.fileo	 = sndfile(self.newname,model=self.filei)
		self.myvec	 = fvec(self.params.hopsize,self.channels)
		self.mycopy	= fvec(self.params.hopsize,self.channels)
		self.slicetimes = slicetimes 

	def __call__(self):
		task.__call__(self)
		# write to current file
		if len(self.slicetimes) and self.frameread >= self.slicetimes[0]:
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
				self.frameread/self.step,".",self.input.split(".")[-1]),model=self.filei)
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

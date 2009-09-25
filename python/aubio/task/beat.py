from aubio.aubioclass import *
from onset import taskonset

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
		self.old       = -1000

	def __call__(self):
		taskonset.__call__(self)
		#results = taskonset.__call__(self)
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
		#if not results: val = 0
		#else: val = results[1] 
		self.dfframe.set(val,self.btwinlen - self.btstep + self.pos2,0)
		i=0
		for i in range(1,int( self.btoutput.get(0,0) ) ):
			if self.pos2 == self.btoutput.get(i,0) and \
				aubio_silence_detection(self.myvec(),
					self.params.silence)!=1: 
				now = self.frameread-0
				period = (60 * self.params.samplerate) / ((now - self.old) * self.params.hopsize)
				self.old = now
				return now,period

	def eval(self,results,tol=0.20,tolcontext=0.25):
		obeats = self.gettruth()
		etime = [result[0] for result in results]
		otime = [obeat[0] for obeat in obeats]
		CML_tot, CML_max, CML_start, CML_end = 0,0,0,0
		AML_tot, AML_max, AML_start, AML_end = 0,0,0,0
		AMLd_tot, AMLd_max, AMLd_start, AMLd_end = 0,0,0,0
		AMLh_tot, AMLh_max, AMLh_start, AMLh_end = 0,0,0,0
		AMLo_tot, AMLo_max, AMLo_start, AMLo_end = 0,0,0,0
		# results iteration
		j = 1
		# for each annotation
		for i in range(2,len(otime)-2):
			if j+1 >= len(etime): break
			count = 0
			# look for next matching beat
			while otime[i] > etime[j] - (otime[i] - otime[i+1])*tol:
				if count > 0: 
					#print "spurious etime"
					if CML_end - CML_start > CML_max:
						CML_max = CML_end - CML_start
					CML_start, CML_end = j, j
					if AMLh_end - AMLh_start > AMLh_max:
						AMLh_max = AMLh_end - AMLh_start
					AMLh_start, AMLh_end = j, j
					if AMLd_end - AMLd_start > AMLd_max:
						AMLd_max = AMLd_end - AMLd_start
					AMLd_start, AMLd_end = j, j
					if AMLo_end - AMLo_start > AMLo_max:
						AMLo_max = AMLo_end - AMLo_start
					AMLo_start, AMLo_end = j, j
				j += 1
				count += 1
			if j+1 >= len(etime): break
			#print otime[i-1],etime[j-1]," ",otime[i],etime[j]," ",otime[i+1],etime[j+1] 
			prevtempo = (otime[i] - otime[i-1])
			nexttempo = (otime[i+1] - otime[i])

			current0  = (etime[j] > otime[i] - prevtempo*tol)
			current1  = (etime[j] < otime[i] + prevtempo*tol)

			# check correct tempo 
			prev0 = (etime[j-1] > otime[i-1] - prevtempo*tolcontext)
			prev1 = (etime[j-1] < otime[i-1] + prevtempo*tolcontext)
			next0 = (etime[j+1] > otime[i+1] - nexttempo*tolcontext)
			next1 = (etime[j+1] < otime[i+1] + nexttempo*tolcontext)

			# check for off beat
			prevoffb0 = (etime[j-1] > otime[i-1] - prevtempo/2 - prevtempo*tolcontext)
			prevoffb1 = (etime[j-1] < otime[i-1] - prevtempo/2 + prevtempo*tolcontext)
			nextoffb0 = (etime[j+1] > otime[i+1] - nexttempo/2 - nexttempo*tolcontext)
			nextoffb1 = (etime[j+1] < otime[i+1] - nexttempo/2 + nexttempo*tolcontext)

			# check half tempo 
			prevhalf0 = (etime[j-1] > otime[i-1] + prevtempo - prevtempo/2*tolcontext)
			prevhalf1 = (etime[j-1] < otime[i-1] + prevtempo + prevtempo/2*tolcontext)
			nexthalf0 = (etime[j+1] > otime[i+1] - nexttempo - nexttempo/2*tolcontext)
			nexthalf1 = (etime[j+1] < otime[i+1] - nexttempo + nexttempo/2*tolcontext)

			# check double tempo
			prevdoub0 = (etime[j-1] > otime[i-1] - prevtempo - prevtempo*2*tolcontext)
			prevdoub1 = (etime[j-1] < otime[i-1] - prevtempo + prevtempo*2*tolcontext)
			nextdoub0 = (etime[j+1] > otime[i+1] + nexttempo - nexttempo*2*tolcontext)
			nextdoub1 = (etime[j+1] < otime[i+1] + nexttempo + nexttempo*2*tolcontext)

			if current0 and current1 and prev0 and prev1 and next0 and next1: 
				#print "YES!"
				CML_end = j	
				CML_tot += 1
			else:
				if CML_end - CML_start > CML_max:
					CML_max = CML_end - CML_start
				CML_start, CML_end = j, j
			if current0 and current1 and prevhalf0 and prevhalf1 and nexthalf0 and nexthalf1: 
				AMLh_end = j
				AMLh_tot += 1
			else:
				if AMLh_end - AMLh_start > AMLh_max:
					AMLh_max = AMLh_end - AMLh_start
				AMLh_start, AMLh_end = j, j
			if current0 and current1 and prevdoub0 and prevdoub1 and nextdoub0 and nextdoub1: 
				AMLd_end = j
				AMLd_tot += 1
			else:
				if AMLd_end - AMLd_start > AMLd_max:
					AMLd_max = AMLd_end - AMLd_start
				AMLd_start, AMLd_end = j, j
			if current0 and current1 and prevoffb0 and prevoffb1 and nextoffb0 and nextoffb1: 
				AMLo_end = j
				AMLo_tot += 1
			else:
				if AMLo_end - AMLo_start > AMLo_max:
					AMLo_max = AMLo_end - AMLo_start
				AMLo_start, AMLo_end = j, j
			# look for next matching beat
			count = 0 
			while otime[i] > etime[j] - (otime[i] - otime[i+1])*tolcontext:
				j += 1
				if count > 0: 
					#print "spurious etime"
					start = j
				count += 1
		total = float(len(otime))
		CML_tot  /= total 
		AMLh_tot /= total 
		AMLd_tot /= total 
		AMLo_tot /= total 
		CML_cont  = CML_max/total
		AMLh_cont = AMLh_max/total
		AMLd_cont = AMLd_max/total
		AMLo_cont = AMLo_max/total
		return CML_cont, CML_tot, AMLh_cont, AMLh_tot, AMLd_cont, AMLd_tot, AMLo_cont, AMLo_tot

#		for i in allfreq:
#			freq.append(float(i) / 2. / N  * samplerate )
#			while freq[i]>freqs[j]:
#				j += 1
#			a0 = weight[j-1]
#			a1 = weight[j]
#			f0 = freqs[j-1]
#			f1 = freqs[j]
#			if f0!=0:
#				iweight.append((a1-a0)/(f1-f0)*freq[i] + (a0 - (a1 - a0)/(f1/f0 -1.)))
#			else:
#				iweight.append((a1-a0)/(f1-f0)*freq[i] + a0)
#			while freq[i]>freqs[j]:
#				j += 1
			
	def eval2(self,results,tol=0.2):
		truth = self.gettruth()
		obeats = [i[0] for i in truth] 
		ebeats = [i[0]*self.params.step for i in results] 
		NP = max(len(obeats), len(ebeats))
		N  = int(round(max(max(obeats), max(ebeats))*100.)+100)
		W  = int(round(tol*100.*60./median([i[1] for i in truth], len(truth)/2)))
		ofunc = [0 for i in range(N+W)]
		efunc = [0 for i in range(N+W)]
		for i in obeats: ofunc[int(round(i*100.)+W)] = 1
		for i in ebeats: efunc[int(round(i*100.)+W)] = 1
		assert len(obeats) == sum(ofunc)
		autocor = 0; m =0
		for m in range (-W, W):
			for i in range(W,N):
				autocor += ofunc[i] * efunc[i-m] 
		autocor /= float(NP)
		return autocor
	
	def evaluation(self,results,tol=0.2,start=5.):

		""" beat tracking evaluation function

		computes P-score of experimental results (ebeats)
		        against ground truth annotations (obeats) """

		from aubio.median import short_find as median
		truth = self.gettruth()
		ebeats = [i[0]*self.params.step for i in results] 
		obeats = [i[0] for i in truth] 

		# trim anything found before start
		while obeats[0] < start: obeats.pop(0)
		while ebeats[0] < start: ebeats.pop(0)
		# maximum number of beats found 
		NP = max(len(obeats), len(ebeats))
		# length of ofunc and efunc vector 
		N  = int(round(max(max(obeats), max(ebeats))*100.)+100)
		# compute W median of ground truth tempi 
		tempi = []
		for i in range(1,len(obeats)): tempi.append(obeats[i]-obeats[i-1])
		W  = int(round(tol*100.*median(tempi,len(tempi)/2)))
		# build ofunc and efunc functions, starting with W zeros  
		ofunc = [0 for i in range(N+W)]
		efunc = [0 for i in range(N+W)]
		for i in obeats: ofunc[int(round(i*100.)+W)] = 1
		for i in ebeats: efunc[int(round(i*100.)+W)] = 1
		# optional: make sure we didn't miss any beats  
		assert len(obeats) == sum(ofunc)
		assert len(ebeats) == sum(efunc)
		# compute auto correlation 
		autocor = 0; m =0
		for m in range (-W, W):
		  for i in range(W,N):
		    autocor += ofunc[i] * efunc[i-m] 
		autocor /= float(NP)
		return autocor

	def gettruth(self):
		import os.path
		from aubio.txtfile import read_datafile
		datafile = self.input.replace('.wav','.txt')
		if not os.path.isfile(datafile):
			print "no ground truth "
			return False,False
		else:
			values = read_datafile(datafile,depth=0)
			old = -1000
			for i in range(len(values)):
				now = values[i]
				period = 60 / (now - old)
				old = now
				values[i] = [now,period]
		return values
	

	def plot(self,oplots,results):
		import Gnuplot
		oplots.append(Gnuplot.Data(results,with_='linespoints',title="auto"))

	def plotplot(self,wplot,oplots,outplot=None,extension=None,xsize=1.,ysize=1.,spectro=False):
		import Gnuplot
		from aubio.gnuplot import gnuplot_create, audio_to_array, make_audio_plot
		import re
		# audio data
		#time,data = audio_to_array(self.input)
		#f = make_audio_plot(time,data)

		g = gnuplot_create(outplot=outplot, extension=extension)
		oplots = [Gnuplot.Data(self.gettruth(),with_='linespoints',title="orig")] + oplots
		g.plot(*oplots)

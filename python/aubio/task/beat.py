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


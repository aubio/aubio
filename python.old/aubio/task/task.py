from aubio.aubioclass import * 
from params import taskparams

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
		self.params.step = float(self.params.hopsize)/float(self.srate)
		self.myvec     = fvec(self.params.hopsize)
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
		#print "CPU time is now %f seconds," % time.clock(),
		#print "task execution took %f seconds" % (time.time() - self.tic)
		return time.time() - self.tic

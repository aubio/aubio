from task import task
from aubio.aubioclass import *

class taskcut(task):
	def __init__(self,input,slicetimes,params=None,output=None):
		""" open the input file and initialize arguments 
		parameters should be set *before* calling this method.
		"""
		from os.path import basename,splitext
		task.__init__(self,input,output=None,params=params)
		self.soundoutbase, self.soundoutext = splitext(basename(self.input))
		self.newname   = "%s%s%09.5f%s%s" % (self.soundoutbase,".",
					self.frameread*self.params.step,".",self.soundoutext)
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
			self.fileo = sndfile("%s%s%09.5f%s%s" % (self.soundoutbase,".",
				self.frameread*self.params.step,".",self.soundoutext),model=self.filei)
			writesize = self.fileo.write(fromcross,self.mycopy)
		else:
			writesize = self.fileo.write(self.readsize,self.myvec)



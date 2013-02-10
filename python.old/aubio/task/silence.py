from aubio.task.task import task
from aubio.aubioclass import *

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




from aubio.aubioclass import aubio_pitchm_freq

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
		self.pitchsmooth = 7
		self.pitchmin=100.
		self.pitchmax=1000.
		self.dcthreshold = -1.
		self.omode = aubio_pitchm_freq
		self.verbose   = False


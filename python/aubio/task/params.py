
class taskparams(object):
	""" default parameters for task classes """
	def __init__(self,input=None,output=None):
		self.silence = -90
		self.derivate = False
		self.localmin = False
		self.delay = 4.
		self.storefunc = False
		self.bufsize = 512
		self.hopsize = 256
		self.pbufsize = 2048
		self.phopsize =  512
		self.samplerate = 44100
		self.tol = 0.05
		self.mintol = 0.0
		self.step = float(self.hopsize)/float(self.samplerate)
		self.threshold = 0.1
		self.onsetmode = 'dual'
		self.pitchmode = 'yin'
		# best threshold for yin monophonic Mirex04 (depth of f0) 
		self.yinthresh = 0.15 
		# best thresh for yinfft monophonic Mirex04 (tradeoff sil/gd)
		# also best param for yinfft polyphonic Mirex04
		self.yinfftthresh = 0.85 
		self.pitchsmooth = 0
		self.pitchmin=20.
		self.pitchmax=20000.
		self.pitchdelay = -0.5
		self.dcthreshold = -1.
		self.omode = "freq"
		self.verbose   = False


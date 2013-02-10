
from aubio.bench.node import *
from os.path import dirname,basename

def mmean(l):
	return sum(l)/max(float(len(l)),1)

def stdev(l):
	smean = 0
	if not len(l): return smean
	lmean = mmean(l)
	for i in l:
		smean += (i-lmean)**2
	smean *= 1. / len(l)
	return smean**.5

class benchonset(bench):

	""" list of values to store per file """
	valuenames = ['orig','missed','Tm','expc','bad','Td']
	""" list of lists to store per file """
	valuelists = ['l','labs']
	""" list of values to print per dir """
	printnames = [ 'mode', 'thres', 'dist', 'prec', 'recl',
		'GD', 'FP', 
		'Torig', 'Ttrue', 'Tfp',  'Tfn',  'TTm',   'TTd',
		'aTtrue', 'aTfp', 'aTfn', 'aTm',  'aTd',  
		'mean', 'smean',  'amean', 'samean']

	""" per dir """
	formats = {'mode': "%12s" , 'thres': "%5.4s", 
		'dist':  "%5.4s", 'prec': "%5.4s", 'recl':  "%5.4s",
		'Torig': "%5.4s", 'Ttrue': "%5.4s", 'Tfp':   "%5.4s", 'Tfn':   "%5.4s", 
		'TTm':    "%5.4s", 'TTd':    "%5.4s",
		'aTtrue':"%5.4s", 'aTfp':  "%5.4s", 'aTfn':  "%5.4s", 
		'aTm':   "%5.4s", 'aTd':   "%5.4s",
		'mean':  "%5.6s", 'smean': "%5.6s", 
		'amean':  "%5.6s", 'samean': "%5.6s", 
		"GD":     "%5.4s", "FP":     "%5.4s",
		"GDm":     "%5.4s", "FPd":     "%5.4s",
		"bufsize": "%5.4s", "hopsize": "%5.4s",
		"time":   "%5.4s"}

	def dir_eval(self):
		""" evaluate statistical data over the directory """
		v = self.v

		v['mode']      = self.params.onsetmode
		v['thres']     = self.params.threshold 
		v['bufsize']   = self.params.bufsize
		v['hopsize']   = self.params.hopsize
		v['silence']   = self.params.silence
		v['mintol']   = self.params.mintol

		v['Torig']     = sum(v['orig'])
		v['TTm']       = sum(v['Tm'])
		v['TTd']       = sum(v['Td'])
		v['Texpc']     = sum(v['expc'])
		v['Tbad']      = sum(v['bad'])
		v['Tmissed']   = sum(v['missed'])
		v['aTm']       = mmean(v['Tm'])
		v['aTd']       = mmean(v['Td'])

		v['mean']      = mmean(v['l'])
		v['smean']     = stdev(v['l'])

		v['amean']     = mmean(v['labs'])
		v['samean']    = stdev(v['labs'])
		
		# old type calculations
		# good detection rate 
		v['GD']  = 100.*(v['Torig']-v['Tmissed']-v['TTm'])/v['Torig']
		# false positive rate
		v['FP']  = 100.*(v['Tbad']+v['TTd'])/v['Torig']
		# good detection counting merged detections as good
		v['GDm'] = 100.*(v['Torig']-v['Tmissed'])/v['Torig'] 
		# false positives counting doubled as good
		v['FPd'] = 100.*v['Tbad']/v['Torig']                
		
		# mirex type annotations
		totaltrue = v['Texpc']-v['Tbad']-v['TTd']
		totalfp = v['Tbad']+v['TTd']
		totalfn = v['Tmissed']+v['TTm']
		self.v['Ttrue']     = totaltrue
		self.v['Tfp']       = totalfp
		self.v['Tfn']       = totalfn
		# average over the number of annotation files
		N = float(len(self.reslist))
		self.v['aTtrue']    = totaltrue/N
		self.v['aTfp']      = totalfp/N
		self.v['aTfn']      = totalfn/N

		# F-measure
		self.P = 100.*float(totaltrue)/max(totaltrue + totalfp,1)
		self.R = 100.*float(totaltrue)/max(totaltrue + totalfn,1)
		#if self.R < 0: self.R = 0
		self.F = 2.* self.P*self.R / max(float(self.P+self.R),1)
		self.v['dist']      = self.F
		self.v['prec']      = self.P
		self.v['recl']      = self.R


	"""
	Plot functions 
	"""

	def plotroc(self,d,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		gd = []
		fp = []
		for i in self.vlist:
			gd.append(i['GD']) 
			fp.append(i['FP']) 
		d.append(Gnuplot.Data(fp, gd, with_='linespoints', 
			title="%s %s" % (plottitle,i['mode']) ))

	def plotplotroc(self,d,outplot=0,extension='ps'):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
			elif extension == 'png': ext, extension = '.png', 'png'
			elif extension == 'svg': ext, extension = '.svg', 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % extension)
			g('set output \'roc-%s%s\'' % (outplot,ext))
		xmax = 30 #max(fp)
		ymin = 50 
		g('set xrange [0:%f]' % xmax)
		g('set yrange [%f:100]' % ymin)
		# grid set
		g('set grid')
		g('set xtics 0,5,%f' % xmax)
		g('set ytics %f,5,100' % ymin)
		g('set key 27,65')
		#g('set format \"%g\"')
		g.title(basename(self.datadir))
		g.xlabel('false positives (%)')
		g.ylabel('correct detections (%)')
		g.plot(*d)

	def plotpr(self,d,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		x = []
		y = []
		for i in self.vlist:
			x.append(i['prec']) 
			y.append(i['recl']) 
		d.append(Gnuplot.Data(x, y, with_='linespoints', 
			title="%s %s" % (plottitle,i['mode']) ))

	def plotplotpr(self,d,outplot=0,extension='ps'):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
			elif extension == 'png': ext, extension = '.png', 'png'
			elif extension == 'svg': ext, extension = '.svg', 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % extension)
			g('set output \'pr-%s%s\'' % (outplot,ext))
		g.title(basename(self.datadir))
		g.xlabel('Recall (%)')
		g.ylabel('Precision (%)')
		g.plot(*d)

	def plotfmeas(self,d,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		x,y = [],[]
		for i in self.vlist:
			x.append(i['thres']) 
			y.append(i['dist']) 
		d.append(Gnuplot.Data(x, y, with_='linespoints', 
			title="%s %s" % (plottitle,i['mode']) ))

	def plotplotfmeas(self,d,outplot="",extension='ps', title="F-measure"):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  terminal = 'postscript'
			elif extension == 'png': terminal = 'png'
			elif extension == 'svg': terminal = 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % terminal)
			g('set output \'fmeas-%s.%s\'' % (outplot,extension))
		g.xlabel('threshold \\delta')
		g.ylabel('F-measure (%)')
		g('set xrange [0:1.2]')
		g('set yrange [0:100]')
		g.title(basename(self.datadir))
		# grid set
		#g('set grid')
		#g('set xtics 0,5,%f' % xmax)
		#g('set ytics %f,5,100' % ymin)
		#g('set key 27,65')
		#g('set format \"%g\"')
		g.plot(*d)

	def plotfmeasvar(self,d,var,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		x,y = [],[]
		for i in self.vlist:
			x.append(i[var]) 
			y.append(i['dist']) 
		d.append(Gnuplot.Data(x, y, with_='linespoints', 
			title="%s %s" % (plottitle,i['mode']) ))
	
	def plotplotfmeasvar(self,d,var,outplot="",extension='ps', title="F-measure"):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  terminal = 'postscript'
			elif extension == 'png': terminal = 'png'
			elif extension == 'svg': terminal = 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % terminal)
			g('set output \'fmeas-%s.%s\'' % (outplot,extension))
		g.xlabel(var)
		g.ylabel('F-measure (%)')
		#g('set xrange [0:1.2]')
		g('set yrange [0:100]')
		g.title(basename(self.datadir))
		g.plot(*d)

	def plotdiffs(self,d,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		v = self.v
		l = v['l']
		mean   = v['mean']
		smean  = v['smean']
		amean  = v['amean']
		samean = v['samean']
		val = []
		per = [0] * 100
		for i in range(0,100):
			val.append(i*.001-.05)
			for j in l: 
				if abs(j-val[i]) <= 0.001:
					per[i] += 1
		total = v['Torig']
		for i in range(len(per)): per[i] /= total/100.

		d.append(Gnuplot.Data(val, per, with_='fsteps', 
			title="%s %s" % (plottitle,v['mode']) ))
		#d.append('mean=%f,sigma=%f,eps(x) title \"\"'% (mean,smean))
		#d.append('mean=%f,sigma=%f,eps(x) title \"\"'% (amean,samean))


	def plotplotdiffs(self,d,outplot=0,extension='ps'):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
			elif extension == 'png': ext, extension = '.png', 'png'
			elif extension == 'svg': ext, extension = '.svg', 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % extension)
			g('set output \'diffhist-%s%s\'' % (outplot,ext))
		g('eps(x) = 1./(sigma*(2.*3.14159)**.5) * exp ( - ( x - mean ) ** 2. / ( 2. * sigma ** 2. ))')
		g.title(basename(self.datadir))
		g.xlabel('delay to hand-labelled onset (s)')
		g.ylabel('% number of correct detections / ms ')
		g('set xrange [-0.05:0.05]')
		g('set yrange [0:20]')
		g.plot(*d)


	def plothistcat(self,d,plottitle=""):
		import Gnuplot, Gnuplot.funcutils
		total = v['Torig']
		for i in range(len(per)): per[i] /= total/100.

		d.append(Gnuplot.Data(val, per, with_='fsteps', 
			title="%s %s" % (plottitle,v['mode']) ))
		#d.append('mean=%f,sigma=%f,eps(x) title \"\"'% (mean,smean))
		#d.append('mean=%f,sigma=%f,eps(x) title \"\"'% (amean,samean))


	def plotplothistcat(self,d,outplot=0,extension='ps'):
		import Gnuplot, Gnuplot.funcutils
		from sys import exit
		g = Gnuplot.Gnuplot(debug=0, persist=1)
		if outplot:
			if   extension == 'ps':  ext, extension = '.ps' , 'postscript'
			elif extension == 'png': ext, extension = '.png', 'png'
			elif extension == 'svg': ext, extension = '.svg', 'svg'
			else: exit("ERR: unknown plot extension")
			g('set terminal %s' % extension)
			g('set output \'diffhist-%s%s\'' % (outplot,ext))
		g('eps(x) = 1./(sigma*(2.*3.14159)**.5) * exp ( - ( x - mean ) ** 2. / ( 2. * sigma ** 2. ))')
		g.title(basename(self.datadir))
		g.xlabel('delay to hand-labelled onset (s)')
		g.ylabel('% number of correct detections / ms ')
		g('set xrange [-0.05:0.05]')
		g('set yrange [0:20]')
		g.plot(*d)



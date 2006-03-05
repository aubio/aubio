from config import *
import commands,sys
import re

def runcommand(cmd,debug=0):
        if VERBOSE >= VERBOSE_CMD or debug: print cmd
        if debug: return 
        status, output = commands.getstatusoutput(cmd)
        if status == 0 or VERBOSE >= VERBOSE_OUT:
                output = output.split('\n')
        if VERBOSE >= VERBOSE_OUT: 
                for i in output: 
                        if i: print i
        if not status == 0: 
                print 'error:',status,output
                print 'command returning error was',cmd
                #sys.exit(1)
	if output == '' or output == ['']: return
        return output 

def list_files(datapath,filter='f', maxdepth = -1):
	if not os.path.exists(datapath):
		print
		print "ERR: no directory %s were found" % datapath
		sys.exit(1)
	if maxdepth >= 0: maxstring = " -maxdepth %d " % maxdepth	
	else: maxstring = ""
        cmd = '%s' * 6 % ('find ',datapath,maxstring,' -type ',filter, "| sort -n")
        return runcommand(cmd)

def list_wav_files(datapath,maxdepth = -1):
	return list_files(datapath, filter="f -name '*.wav'",maxdepth = maxdepth)

sndfile_filter = "f -name '*.wav' -o -name '*.aif' -o -name '*.aiff'"

def list_snd_files(datapath,maxdepth = -1):
	return list_files(datapath, filter=sndfile_filter, 
		maxdepth = maxdepth)

def list_res_files(datapath,maxdepth = -1):
	return list_files(datapath, filter="f -name '*.txt'", maxdepth = maxdepth)

def list_dirs(datapath):
	return list_files(datapath, filter="d")

def mkdir(path):
        cmd = '%s%s' % ('mkdir -p ',path)
        return runcommand(cmd)

def act_on_data (action,datapath,respath=None,suffix='.txt',filter='f',sub='\.wav$',**keywords):
        """ execute action(datafile,resfile) on all files in datapath """
        dirlist = list_files(datapath,filter=filter)
        if dirlist == ['']: dirlist = []
        if respath:
		respath_in_datapath = re.split(datapath, respath,maxsplit=1)[1:]
        	if(respath_in_datapath and suffix == ''): 
                	print 'error: respath in datapath and no suffix used'
        for i in dirlist:
                j = re.split(datapath, i,maxsplit=1)[1]
                j = re.sub(sub,'',j)
                #j = "%s%s%s"%(respath,j,suffix)
		if respath:
			j = "%s%s"%(respath,j)
			if sub != '':
				j = re.sub(sub,suffix,j)
			else:
				j = "%s%s" % (j,suffix)
                action(i,j,**keywords)

def act_on_results (action,datapath,respath,filter='d'):
        """ execute action(respath) an all subdirectories in respath """
        dirlist = list_files(datapath,filter='d')
        respath_in_datapath = re.split(datapath, respath,maxsplit=1)[1:]
        if(respath_in_datapath and not filter == 'd' and suffix == ''): 
                print 'warning: respath is in datapath'
        for i in dirlist:
                s = re.split(datapath, i ,maxsplit=1)[1]
                action("%s%s%s"%(respath,'/',s))

def act_on_files (action,listfiles,listres=None,suffix='.txt',filter='f',sub='\.wav$',**keywords):
        """ execute action(respath) an all subdirectories in respath """
        if listres and len(listfiles) <= len(listres): 
		for i in range(len(listfiles)):
			action(listfiles[i],listres[i],**keywords)
        else:
		for i in listfiles:
                	action(i,None,**keywords)

class bench:
	""" class to run benchmarks on directories """
	def __init__(self,datadir,resdir=None,checkres=False,checkanno=False,params=[]):
		from aubio.task.params import taskparams
		self.datadir = datadir
		# path to write results path to
		self.resdir = resdir
		# list of annotation files
		self.reslist = []
		# list used to gather results
		self.results = []
		if not params: self.params = taskparams()
		else:          self.params = params
		print "Checking data directory", self.datadir
		self.checkdata()
		if checkanno: self.checkanno()
		if checkres: self.checkres()
	
	def checkdata(self):
		if os.path.isfile(self.datadir):
			self.dirlist = os.path.dirname(self.datadir)
		elif os.path.isdir(self.datadir):
			self.dirlist = list_dirs(self.datadir)
		# allow dir* matching through find commands?
		else:
			print "ERR: path not understood"
			sys.exit(1)
		print "Listing directories in data directory",
		if self.dirlist:
			print " (%d elements)" % len(self.dirlist)
		else:
			print " (0 elements)"
			print "ERR: no directory %s were found" % self.datadir
			sys.exit(1)
		print "Listing sound files in data directory",
		self.sndlist = list_snd_files(self.datadir)
		if self.sndlist:
			print " (%d elements)" % len(self.sndlist)
		else:
			print " (0 elements)"
			print "ERR: no sound files were found in", self.datadir
			sys.exit(1)
	
	def checkanno(self):
		print "Listing annotations in data directory",
		self.reslist = list_res_files(self.datadir)
		print " (%d elements)" % len(self.reslist)
		#for each in self.reslist: print each
		if not self.reslist or len(self.reslist) < len (self.sndlist):
			print "ERR: not enough annotations"
			return -1
		else:
			print "Found enough annotations"
	
	def checkres(self):
		print "Creating results directory"
		act_on_results(mkdir,self.datadir,self.resdir,filter='d')

	def pretty_print(self,sep='|'):
		for i in self.printnames:
			print self.formats[i] % self.v[i], sep,
		print

	def pretty_titles(self,sep='|'):
		for i in self.printnames:
			print self.formats[i] % i, sep,
		print

	def dir_exec(self):
		""" run file_exec on every input file """
		self.l , self.labs = [], [] 
		self.v = {}
		for i in self.valuenames:
			self.v[i] = [] 
		for i in self.valuelists:
			self.v[i] = [] 
		act_on_files(self.file_exec,self.sndlist,self.reslist, \
			suffix='',filter=sndfile_filter)

	def dir_eval(self):
		pass

	def file_gettruth(self,input):
		""" get ground truth filenames """
		from os.path import isfile
		ftrulist = []
		# search for match as filetask.input,".txt" 
		ftru = '.'.join(input.split('.')[:-1])
		ftru = '.'.join((ftru,'txt'))
		if isfile(ftru):
			ftrulist.append(ftru)
		else:
			# search for matches for filetask.input in the list of results
			for i in range(len(self.reslist)):
				check = '.'.join(self.reslist[i].split('.')[:-1])
				check = '_'.join(check.split('_')[:-1])
				if check == '.'.join(input.split('.')[:-1]):
					ftrulist.append(self.reslist[i])
		return ftrulist

	def file_exec(self,input,output):
		""" create filetask, extract data, evaluate """
		filetask = self.task(input,params=self.params)
		computed_data = filetask.compute_all()
		ftrulist = self.file_gettruth(filetask.input)
		for i in ftrulist:
			filetask.eval(computed_data,i,mode='rocloc',vmode='')
			""" append filetask.v to self.v """
			for i in self.valuenames:
				self.v[i].append(filetask.v[i])
			for j in self.valuelists:
				if filetask.v[j]:
					for i in range(len(filetask.v[j])):
						self.v[j].append(filetask.v[j][i])
	
	def file_eval(self):
		pass
	
	def file_plot(self):
		pass

	def dir_plot(self):
		pass
	
	def run_bench(self):
		for mode in self.modes:
			self.params.mode = mode
			self.dir_exec()
			self.dir_eval()
			self.dir_plot()

	def dir_eval_print(self):
		self.dir_exec()
		self.dir_eval()
		self.pretty_print()


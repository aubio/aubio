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
        cmd = '%s' * 5 % ('find ',datapath,maxstring,' -type ',filter)
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

class bench:
	""" class to run benchmarks on directories """
	def __init__(self,datadir,resdir=None,checkres=False,checkanno=False):
		self.datadir = datadir
		self.resdir = resdir
		self.results = []
		print "Checking data directory", self.datadir
		self.checkdata()
		if checkanno: self.checkanno()
		if checkres: self.checkres()
	
	def checkdata(self):
		print "Listing directories in data directory",
		self.dirlist = list_dirs(self.datadir)
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
		#for each in self.sndlist: print each
	
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

	def pretty_print(self,values):
		for i in range(len(values)):
			print self.formats[i] % values[i],
		print

	def dir_exec(self):
		""" run file_exec on every input file """
		self.orig, self.missed, self.merged, self.expc, \
			self.bad, self.doubled = 0, 0, 0, 0, 0, 0
		act_on_data(self.file_exec,self.datadir,self.resdir, \
			suffix='',filter=sndfile_filter)
	
	def dir_eval(self):
		pass

	def file_exec(self):
		pass
	
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

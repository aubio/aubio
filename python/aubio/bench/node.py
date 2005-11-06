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
        return output 

def list_files(datapath,filter='f'):
        cmd = '%s%s%s%s' % ('find ',datapath,' -type ',filter)
        return runcommand(cmd)

def list_wav_files(datapath,maxdepth=1):
	return list_files(datapath, filter="f -name '*.wav' -maxdepth %d"%maxdepth)

def list_dirs(datapath):
	return list_files(datapath, filter="d")

def mkdir(path):
        cmd = '%s%s' % ('mkdir -p ',path)
        return runcommand(cmd)

def act_on_data (action,datapath,respath,suffix='.txt',filter='f',sub='\.wav$',**keywords):
        """ execute action(datafile,resfile) on all files in datapath """
        dirlist = list_files(datapath,filter=filter)
        if dirlist == ['']: dirlist = []
        respath_in_datapath = re.split(datapath, respath,maxsplit=1)[1:]
        if(respath_in_datapath and suffix == ''): 
                print 'error: respath in datapath and no suffix used'
                #sys.exit(1)
        for i in dirlist:
                j = re.split(datapath, i,maxsplit=1)[1]
                j = re.sub(sub,'',j)
                #j = "%s%s%s"%(respath,j,suffix)
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

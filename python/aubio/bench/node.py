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
                sys.exit(1)
        return output 

def list_files(datapath,filter='f'):
        cmd = '%s%s%s%s' % ('find ',datapath,' -type ',filter)
        return runcommand(cmd)

def mkdir(path):
        cmd = '%s%s' % ('mkdir -p ',path)
        return runcommand(cmd)

def act_on_data (action,datapath,respath,suffix='.txt',filter='f'):
        """ execute action(datafile,resfile) on all files in datapath """
        dirlist = list_files(datapath,filter=filter)
        respath_in_datapath = re.split(datapath, respath,maxsplit=1)[1:]
        if(respath_in_datapath and suffix == ''): 
                print 'error: respath in datapath and no suffix used'
                sys.exit(1)
        for i in dirlist:
                s = re.split(datapath, i,maxsplit=1)[1]
                j = "%s%s%s%s"%(respath,'/',s,suffix)
                action(i,j)

def act_on_results (action,datapath,respath,filter='d'):
        """ execute action(respath) an all subdirectories in respath """
        dirlist = list_files(datapath,filter='d')
        respath_in_datapath = re.split(datapath, respath,maxsplit=1)[1:]
        if(respath_in_datapath and not filter == 'd' and suffix == ''): 
                print 'warning: respath is in datapath'
        for i in dirlist:
                s = re.split(datapath, i ,maxsplit=1)[1]
                action("%s%s%s"%(respath,'/',s))

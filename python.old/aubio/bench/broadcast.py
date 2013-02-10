from config import *

class run_broadcast:
        def __init__(self,command,*args):
                for host in REMOTEHOSTS:
                        command(host,args[0],args[1:])

def remote_sync(host,path='',options=''):
        optstring = ''
        for i in options:
                optstring = "%s %s" % (optstring,i)
        print RSYNC_CMD,optstring,RSYNC_OPT,' --delete', 
        print '%s%s%s%s%s' % (path,'/ ',host,':',path)


def fetch_results(host,path='',options=''):
        optstring = ''
        for i in options:
                optstring = "%s %s" % (optstring,i)
        print RSYNC_CMD,optstring,RSYNC_OPT,' --update', 
        print '%s%s%s%s%s' % (host,':',path,'/ ',path)

def remote_queue(host,command,options=''):
        print 'oarsub -p "hostname = \'',host,'\'',command
        

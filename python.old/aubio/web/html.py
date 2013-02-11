from aubio.bench.node import *

def parse_args(req):
    req.basehref = BASEHREF
    req.datadir = DATADIR
    if req.path_info: path_info = req.path_info
    else: path_info = '/'
    location = re.sub('^/show_[a-z0-9]*/','',path_info)
    location = re.sub('^/play_[a-z0-9]*/','',location)
    location = re.sub('^/index/','',location)
    location = re.sub('^/','',location)
    location = re.sub('/$','',location)
    datapath = "%s/%s" % (DATADIR,location)
    respath  = "%s/%s" % (DATADIR,location)
    last     = re.sub('/$','',location)
    last     = last.split('/')[-1]
    first    = path_info.split('/')[1]
    # store some of this in the mp_request
    req.location, req.datapath, req.respath = location, datapath, respath
    req.first, req.last = first, last

    if location:
        if not (os.path.isfile(datapath) or 
		os.path.isdir(datapath) or 
		location in ['feedback','email']):
		# the path was not understood
		from mod_python import apache
		req.write("<html> path not found %s</html>" % (datapath))
		raise apache.SERVER_RETURN, apache.OK
		#from mod_python import apache
		#raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND

def navigation(req):
    """ main html navigation header """
    from mod_python import psp
    req.content_type = "text/html"
    parse_args(req)
    datapath = req.datapath
    location = req.location

    # deal with session
    if req.sess.is_new():
	    msg = "<b>Welcome %s</b><br>" % req.sess['login']
    else:
	    msg = "<b>Welcome back %s</b><br>" % req.sess['login']

    # start writing
    tmpl = psp.PSP(req, filename='header.tmpl')
    tmpl.run(vars = { 'title': "aubioweb / %s / %s" % (req.first,location),
    		'basehref': '/~piem/',
		'message': msg,
    		'action': req.first})

    req.write("<h2>Content of ")
    print_link(req,"","/")
    y = location.split('/')
    for i in range(len(y)-1): 
    	print_link(req,"/".join(y[:i+1]),y[i])
	req.write(" / ")
    req.write("%s</h2>\n" % y[-1])

    a = {'show_info' : 'info',
    	 'show_sound': 'waveform',
    	 'show_onset': 'onset',
    	 'index'     : 'index',
	 'show_pitch': 'pitch',
	 'play_m3u': 'stream (m3u/ogg)',
	 'play_ogg': 'save (ogg)',
	 'play_wav': 'save (wav)',
	 }

    # print task lists (only remaining tasks)
    print_link(req,re.sub('%s.*'%req.last,'',location),"go up")
    akeys = a.keys(); akeys.sort();
    curkey = req.first
    for akey in akeys: 
        if akey != curkey:
    		req.write(":: ")
		print_link(req,"/".join((akey,location)),a[akey])
	else:
    		req.write(":: ")
		req.write("<b>%s</b>" % a[akey])
    req.write("<br>")

    # list the content of the directories
    listdir,listfiles = [],[]
    if os.path.isdir(datapath):
        listfiles = list_snd_files(datapath)
    	listdir = list_dirs(datapath)
	listdir.pop(0) # kick the current dir
    elif os.path.isfile(datapath):
        listfiles = [datapath]
	listdir = [re.sub(req.last,'',location)]

    link_list(req,listdir,title="Subdirectories")
    link_list(req,listfiles,title="Files")

def footer(req):
    """ html navigation footer """
    from mod_python import psp
    tmpl = psp.PSP(req, filename='footer.tmpl')
    tmpl.run(vars = { 'time': -req.mtime+req.request_time })

def apply_on_data(req, func,**keywords):
    # bug: hardcoded snd file filter
    act_on_data(func,req.datapath,req.respath,
    	filter="f  -maxdepth 1 -name '*.wav' -o -name '*.aif'",**keywords)

def print_link(req,target,name,basehref=BASEHREF):
    req.write("<a href='%s/%s'>%s</a>\n" % (basehref,target,name))

def print_img(req,target,name='',basehref=BASEHREF):
    if name == '': name = target
    req.write("<img src='%s/%s' alt='%s' title='%s'>\n" % (basehref,target,name,name))

def link_list(req,targetlist,basehref=BASEHREF,title=None):
    if len(targetlist) > 1:
        if title: req.write("<h3>%s</h3>"%title)
        req.write('<ul>')
        for i in targetlist:
            s = re.split('%s/'%DATADIR,i,maxsplit=1)[1]
            if s: 
        	req.write('<li>')
	    	print_link(req,s,s)
        	req.write('</li>')
        req.write('</ul>')

def print_list(req,list):
    req.write("<pre>\n")
    for i in list: req.write("%s\n" % i)
    req.write("</pre>\n")

def print_command(req,command):
    req.write("<h4>%s</h4>\n" % re.sub('%%','%',command))
    def print_runcommand(input,output):
        cmd = re.sub('(%)?%i','%s' % input, command)
        cmd = re.sub('(%)?%o','%s' % output, cmd)
        print_list(req,runcommand(cmd))
    apply_on_data(req,print_runcommand)

def datapath_to_location(input):
    location = re.sub(DATADIR,'',input)
    return re.sub('^/*','',location)

## drawing hacks
def draw_func(req,func):
    import re
    req.content_type = "image/png"
    # build location (strip the func_path, add DATADIR)
    location = re.sub('^/draw_[a-z]*/','%s/'%DATADIR,req.path_info)
    location = re.sub('.png$','',location)
    if not os.path.isfile(location):
	from mod_python import apache
	raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND
    # replace location in func
    cmd = re.sub('(%)?%i','%s' % location, func)
    # add PYTHONPATH at the beginning, 
    cmd = "%s%s 2> /dev/null" % (PYTHONPATH,cmd)
    for each in runcommand(cmd):
	req.write("%s\n"%each)

def show_task(req,task):
    def show_task_file(input,output,task):
        location = datapath_to_location(input)
        print_img(req,"draw_%s/%s" % (task,location))
    navigation(req)
    req.write("<h3>%s</h3>\n" % task)
    apply_on_data(req,show_task_file,task=task)
    footer(req)

## waveform_foo
def draw_sound(req):
    draw_func(req,"aubioplot-audio %%i stdout 2> /dev/null")

def show_sound(req):
    show_task(req,"sound")

## pitch foo
def draw_pitch(req,threshold='0.3'):
    draw_func(req,"aubiopitch -i %%i -p -m schmitt,yin,fcomb,mcomb -t %s -O stdout" % threshold)

def show_pitch(req):
    show_task(req,"pitch")

## onset foo
def draw_onset(req,threshold='0.3'):
    draw_func(req,"aubiocut -i %%i -p -m complex -t %s -O stdout" % threshold)

def show_onset(req,threshold='0.3',details=''):
    def onset_file(input,output):
        location = datapath_to_location(input)
        print_img(req,"draw_onset/%s?threshold=%s"%(location,threshold))
        print_link(req,"?threshold=%s" % (float(threshold)-0.1),"-")
        req.write("%s\n" % threshold)
        print_link(req,"?threshold=%s" % (float(threshold)+0.1),"+")
	# bug: hardcoded sndfile extension 
        anote = re.sub('\.wav$','.txt',input)
	if anote == input: anote = ""
        res = get_extract(input,threshold)
        if os.path.isfile(anote):
            tru = get_anote(anote)
            print_list(req,get_results(tru,res,0.05))
        else:
            req.write("no ground truth found<br>\n")
        if details:
            req.write("<h4>Extraction</h4>\n")
            print_list(req,res)
        else:
            req.write("<a href='%s/show_onset/%s?details=yes&amp;threshold=%s'>details</a><br>\n" %
            	(req.basehref,location,threshold))
        if details and os.path.isfile(anote):
            req.write("<h4>Computed differences</h4>\n")
            ldiffs = get_diffs(tru,res,0.05)
            print_list(req,ldiffs)
            req.write("<h4>Annotations</h4>\n")
            print_list(req,tru)
    navigation(req)
    req.write("<h3>Onset</h3>\n")
    apply_on_data(req,onset_file)
    footer(req)

def get_anote(anote):
    import aubio.onsetcompare
    # FIXME: should import with txtfile.read_datafile
    return aubio.onsetcompare.load_onsets(anote)

def get_diffs(anote,extract,tol):
    import aubio.onsetcompare
    return aubio.onsetcompare.onset_diffs(anote,extract,tol)

def get_extract(datapath,threshold='0.3'):
    cmd = "%saubiocut -v -m complex -t %s -i %s" % (PYTHONPATH,threshold,datapath)
    lo = runcommand(cmd)
    for i in range(len(lo)): lo[i] = float(lo[i])
    return lo

def get_results(anote,extract,tol):
    import aubio.onsetcompare
    orig, missed, merged, expc, bad, doubled = aubio.onsetcompare.onset_roc(anote,extract,tol)
    s =("GD %2.8f\t"        % (100*float(orig-missed-merged)/(orig)),
        "FP %2.8f\t"        % (100*float(bad+doubled)/(orig))       , 
        "GD-merged %2.8f\t" % (100*float(orig-missed)/(orig))       , 
        "FP-pruned %2.8f\t" % (100*float(bad)/(orig))		    )
    return s

# play m3u foo
def play_m3u(req):
    def show_task_file(input,output,task):
        location = datapath_to_location(input)
        req.write("http://%s%s/play_ogg/%s\n" % (HOSTNAME,BASEHREF,re.sub("play_m3u",task,location)))
    req.content_type = "audio/mpegurl"
    parse_args(req)
    apply_on_data(req,show_task_file,task="play_ogg")

# play wav foo
def play_wav(req):
    req.content_type = "audio/x-wav"
    func = "cat %%i"
    # build location (strip the func_path, add DATADIR)
    location = re.sub('^/play_wav/','%s/'%DATADIR,req.path_info)
    if not os.path.isfile(location):
	from mod_python import apache
	raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND
    # replace location in func
    cmd = re.sub('(%)?%i','%s' % location, func)
    # add PYTHONPATH at the beginning, 
    cmd = "%s 2> /dev/null" % cmd
    for each in runcommand(cmd):
	req.write("%s\n"%each)

# play ogg foo
def play_ogg(req):
    req.content_type = "application/ogg"
    func = "oggenc -o - %%i"
    # build location (strip the func_path, add DATADIR)
    location = re.sub('^/play_ogg/','%s/'%DATADIR,req.path_info)
    location = re.sub('.ogg$','',location)
    if not os.path.isfile(location):
	from mod_python import apache
	raise apache.SERVER_RETURN, apache.HTTP_NOT_FOUND
    # replace location in func
    cmd = re.sub('(%)?%i','%s' % location, func)
    # add PYTHONPATH at the beginning, 
    cmd = "%s 2> /dev/null" % cmd
    for each in runcommand(cmd):
	req.write("%s\n"%each)

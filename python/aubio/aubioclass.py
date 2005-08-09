from aubiowrapper import *

class fvec:
    def __init__(self,size,chan):
        self.vec = new_fvec(size,chan)
    def __call__(self):
        return self.vec
    def __del__(self):
        del_fvec(self())
    def get(self,pos,chan):
        return fvec_read_sample(self(),chan,pos)
    def set(self,value,pos,chan):
        return fvec_write_sample(self(),value,chan,pos)
    def channel(self,chan):
        return fvec_get_channel(self(),chan)
    def data(self):
        return fvec_get_data(self())

class cvec:
    def __init__(self,size,chan):
        self.vec = new_cvec(size,chan)
    def __call__(self):
        return self.vec
    def __del__(self):
        del_cvec(self())
    def get(self,pos,chan):
        return fvec_read_sample(self(),chan,pos)

class sndfile:
    def __init__(self,filename,model=None):
        if (model!=None):
            self.file = new_aubio_sndfile_wo(model.file,filename)
        else:
            self.file = new_aubio_sndfile_ro(filename)
    def __del__(self):
        del_aubio_sndfile(self.file)
    def info(self):
        aubio_sndfile_info(self.file)
    def samplerate(self):
        return aubio_sndfile_samplerate(self.file)
    def channels(self):
        return aubio_sndfile_channels(self.file)
    def read(self,nfram,vecread):
        return aubio_sndfile_read(self.file,nfram,vecread())
    def write(self,nfram,vecwrite):
        return aubio_sndfile_write(self.file,nfram,vecwrite())

class pvoc:
    def __init__(self,buf,hop,chan):
        self.pv = new_aubio_pvoc(buf,hop,chan)
    def __del__(self):
        del_aubio_pvoc(self.pv)
    def do(self,tf,tc):
        aubio_pvoc_do(self.pv,tf(),tc())
    def rdo(self,tc,tf):
        aubio_pvoc_rdo(self.pv,tc(),tf())

class onsetdetection:
    """ class for aubio_onsetdetection """
    def __init__(self,type,buf,chan):
        self.od = new_aubio_onsetdetection(type,buf,chan)
    def do(self,tc,tf):
        aubio_onsetdetection(self.od,tc(),tf())
    def __del__(self):
        aubio_onsetdetection_free(self.od)

class peakpick:
    """ class for aubio_peakpicker """
    def __init__(self,threshold=0.1):
        self.pp = new_aubio_peakpicker(threshold)
    def do(self,fv):
        return aubio_peakpick_pimrt(fv(),self.pp)
    def __del__(self):
        del_aubio_peakpicker(self.pp)

class onsetpick:
    """ superclass for aubio_pvoc + aubio_onsetdetection + aubio_peakpicker """
    def __init__(self,bufsize,hopsize,channels,myvec,threshold,mode='dual',derivate=False):
        self.myfft    = cvec(bufsize,channels)
        self.pv       = pvoc(bufsize,hopsize,channels)
        if mode in ['dual'] :
                self.myod     = onsetdetection(hfc,bufsize,channels)
                self.myod2    = onsetdetection(complexdomain,bufsize,channels)
                self.myonset  = fvec(1,channels)
                self.myonset2 = fvec(1,channels)
        else: 
                self.myod     = onsetdetection(mode,bufsize,channels)
                self.myonset  = fvec(1,channels)
        self.mode     = mode
        self.pp       = peakpick(float(threshold))
        self.derivate = derivate
        self.oldval   = 0.

    def do(self,myvec): 
        self.pv.do(myvec,self.myfft)
        self.myod.do(self.myfft,self.myonset)
        if self.mode == 'dual':
                self.myod2.do(self.myfft,self.myonset2)
                self.myonset.set(self.myonset.get(0,0)*self.myonset2.get(0,0),0,0)
        if self.derivate:
                val         = self.myonset.get(0,0)
                dval        = val - self.oldval
                self.oldval = val
                if dval > 0: self.myonset.set(dval,0,0)
                else:  self.myonset.set(0.,0,0)
        return self.pp.do(self.myonset),self.myonset.get(0,0)

def getonsets(filein,threshold=0.2,silence=-70.,bufsize=1024,hopsize=512,
                mode='dual',localmin=False,storefunc=False,derivate=False):
        frameread = 0
        filei     = sndfile(filein)
        channels  = filei.channels()
        myvec     = fvec(hopsize,channels)
        readsize  = filei.read(hopsize,myvec)
        opick     = onsetpick(bufsize,hopsize,channels,myvec,threshold,
                         mode=mode,derivate=derivate)
        mylist    = list()
        if localmin:
                ovalist   = [0., 0., 0., 0., 0.]
        if storefunc:
                ofunclist = []
        while(readsize):
                readsize = filei.read(hopsize,myvec)
                isonset,val = opick.do(myvec)
                if (aubio_silence_detection(myvec(),silence)):
                        isonset=0
                if localmin:
                        if val > 0: ovalist.append(val)
                        else: ovalist.append(0)
                        ovalist.pop(0)
                if storefunc:
                        ofunclist.append(val)
                if (isonset == 1):
                        if localmin:
                                i=len(ovalist)-1
                                # find local minima before peak 
                                while ovalist[i-1] < ovalist[i] and i > 0:
                                        i -= 1
                                now = (frameread+1-i)
                        else:
                                now = frameread
                        if now > 0 :
                                mylist.append(now)
                        else:
                                now = 0
                                mylist.append(now)
                frameread += 1
        if storefunc: return mylist, ofunclist
        else: return mylist

def cutfile(filein,slicetimes,zerothres=0.008,bufsize=1024,hopsize=512):
    frameread = 0
    readsize  = hopsize 
    filei     = sndfile(filein)
    framestep = hopsize/(filei.samplerate()+0.)
    channels  = filei.channels()
    newname   = "%s%f%s" % ("/tmp/",0.0000000,filein[-4:])
    fileo     = sndfile(newname,model=filei)
    myvec     = fvec(hopsize,channels)
    mycopy    = fvec(hopsize,channels)
    while(readsize==hopsize):
        readsize = filei.read(hopsize,myvec)
        # write to current file
        if len(slicetimes) and frameread >= slicetimes[0]:
            slicetimes.pop(0)
            # write up to 1st zero crossing
            zerocross = 0
            while ( abs( myvec.get(zerocross,0) ) > zerothres ):
                zerocross += 1
            writesize = fileo.write(zerocross,myvec)
            fromcross = 0
            while (zerocross < readsize):
                for i in range(channels):
                    mycopy.set(myvec.get(zerocross,i),fromcross,i)
                    fromcross += 1
                    zerocross += 1
            del fileo
            fileo = sndfile("%s%s%f%s%s" % 
                (filein.split(".")[0].split("/")[-1],".",
                frameread*framestep,".",filein.split(".")[-1]),model=filei)
            writesize = fileo.write(fromcross,mycopy)
        else:
            writesize = fileo.write(readsize,myvec)
        frameread += 1
    del fileo


def getsilences(filein,hopsize=512,silence=-70):
    frameread = 0
    filei     = sndfile(filein)
    srate     = filei.samplerate()
    channels  = filei.channels()
    myvec     = fvec(hopsize,channels)
    readsize  = filei.read(hopsize,myvec)
    mylist    = []
    wassilence = 0
    while(readsize==hopsize):
        readsize = filei.read(hopsize,myvec)
        if (aubio_silence_detection(myvec(),silence)==1):
            if wassilence == 0:
                mylist.append(frameread)
                wassilence == 1
        else: wassilence = 0
        frameread += 1
    return mylist

def getpitch(filein,mode=aubio_pitch_mcomb,bufsize=1024,hopsize=512,omode=aubio_pitchm_freq,
        samplerate=44100.,silence=-70):
    frameread = 0
    filei     = sndfile(filein)
    srate     = filei.samplerate()
    channels  = filei.channels()
    myvec     = fvec(hopsize,channels)
    readsize  = filei.read(hopsize,myvec)
    pitchdet  = pitchdetection(mode=mode,bufsize=bufsize,hopsize=hopsize,
                         channels=channels,samplerate=srate,omode=omode)
    mylist    = []
    while(readsize==hopsize):
        readsize = filei.read(hopsize,myvec)
        freq = pitchdet(myvec)
        #print "%.3f     %.2f" % (now,freq)
        if (aubio_silence_detection(myvec(),silence)!=1):
                mylist.append(freq)
        else: 
                mylist.append(0)
        frameread += 1
    return mylist

class pitchdetection:
    def __init__(self,mode=aubio_pitch_mcomb,bufsize=2048,hopsize=1024,
        channels=1,samplerate=44100.,omode=aubio_pitchm_freq):
        self.pitchp = new_aubio_pitchdetection(bufsize,hopsize,channels,
                samplerate,mode,omode)
        #self.filt     = filter(srate,"adsgn")
    def __del__(self):
        del_aubio_pitchdetection(self.pitchp)
    def __call__(self,myvec): 
        #self.filt(myvec)
        return aubio_pitchdetection(self.pitchp,myvec())

class filter:
    def __init__(self,srate,type=None):
        if (type=="adsgn"):
            self.filter = new_aubio_adsgn_filter(srate)
    def __del__(self):
        #del_aubio_filter(self.filter)
        pass
    def __call__(self,myvec):
        aubio_filter_do(self.filter,myvec())

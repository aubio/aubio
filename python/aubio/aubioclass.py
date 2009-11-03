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
        return self.get_norm(pos,chan)
    def set(self,val,pos,chan):
        self.set_norm(val,chan,pos)
    def get_norm(self,pos,chan):
        return cvec_read_norm(self(),chan,pos)
    def set_norm(self,val,pos,chan):
        cvec_write_norm(self(),val,chan,pos)
    def get_phas(self,pos,chan):
        return cvec_read_phas(self(),chan,pos)
    def set_phas(self,val,pos,chan):
        cvec_write_phas(self(),val,chan,pos)

class sndfile:
    def __init__(self,filename,model=None):
        if (model!=None):
            self.file = new_aubio_sndfile_wo(model.file,filename)
        else:
            self.file = new_aubio_sndfile_ro(filename)
        if self.file == None:
            raise IOError, "failed opening file %s" % filename
    def __del__(self):
        if self.file != None: del_aubio_sndfile(self.file)
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
    """ class for aubio_specdesc """
    def __init__(self,mode,buf,chan):
        self.od = new_aubio_specdesc(mode,buf,chan)
    def do(self,tc,tf):
        aubio_specdesc_do(self.od,tc(),tf())
    def __del__(self):
        del_aubio_specdesc(self.od)

class peakpick:
    """ class for aubio_peakpicker """
    def __init__(self,threshold=0.1):
        self.pp = new_aubio_peakpicker(1)
        self.out = new_fvec(1, 1)
        aubio_peakpicker_set_threshold (self.pp, threshold)
    def do(self,fv):
        aubio_peakpicker_do(self.pp, fv(), self.out)
        return fvec_read_sample(self.out, 0, 0)
    def getval(self):
        return aubio_peakpicker_get_adaptive_threshold(self.pp)
    def __del__(self):
        del_aubio_peakpicker(self.pp)

class onsetpick:
    """ superclass for aubio_pvoc + aubio_specdesc + aubio_peakpicker """
    def __init__(self,bufsize,hopsize,channels,myvec,threshold,mode='dual',derivate=False,dcthreshold=0):
        self.myfft    = cvec(bufsize,channels)
        self.pv       = pvoc(bufsize,hopsize,channels)
        if mode in ['dual'] :
                self.myod     = onsetdetection("hfc",bufsize,channels)
                self.myod2    = onsetdetection("mkl",bufsize,channels)
                self.myonset  = fvec(1,channels)
                self.myonset2 = fvec(1,channels)
        else: 
                self.myod     = onsetdetection(mode,bufsize,channels)
                self.myonset  = fvec(1,channels)
        self.mode     = mode
        self.pp       = peakpick(float(threshold))
        self.derivate = derivate
        self.dcthreshold = dcthreshold 
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
        isonset, dval = self.pp.do(self.myonset),self.myonset.get(0,0)
        if self.dcthreshold:
           if dval < self.dcthreshold: isonset = 0 
        return isonset, dval

class pitch:
    def __init__(self,mode="mcomb",bufsize=2048,hopsize=1024,
        channels=1,samplerate=44100.,omode="freq",tolerance=0.1):
        self.pitchp = new_aubio_pitch(mode,bufsize,hopsize,channels,
            samplerate)
        self.mypitch = fvec(1, channels)
        aubio_pitch_set_unit(self.pitchp,omode)
        aubio_pitch_set_tolerance(self.pitchp,tolerance)
        #self.filt     = filter(srate,"adsgn")
    def __del__(self):
        del_aubio_pitch(self.pitchp)
    def __call__(self,myvec): 
        aubio_pitch_do(self.pitchp,myvec(), self.mypitch())
        return self.mypitch.get(0,0)

class filter:
    def __init__(self,srate,type=None):
        if (type=="adsgn"):
            self.filter = new_aubio_adsgn_filter(srate)
    def __del__(self):
        #del_aubio_filter(self.filter)
        pass
    def __call__(self,myvec):
        aubio_filter_do(self.filter,myvec())

class beattracking:
    """ class for aubio_beattracking """
    def __init__(self,winlen,channels):
        self.p = new_aubio_beattracking(winlen,channels)
    def do(self,dfframe,out):
        return aubio_beattracking_do(self.p,dfframe(),out())
    def __del__(self):
        del_aubio_beattracking(self.p)


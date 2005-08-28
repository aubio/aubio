from aubioclass import * 

def check_onset_mode(option, opt, value, parser):
        """ utility function to convert a string to aubio_onsetdetection_type """
        nvalues = parser.rargs[0].split(',')
        val =  []
        for nvalue in nvalues:
                if   nvalue == 'complexdomain' or nvalue == 'complex' :
                         val.append(aubio_onset_complex)
                elif nvalue == 'hfc'           :
                         val.append(aubio_onset_hfc)
                elif nvalue == 'phase'         :
                         val.append(aubio_onset_phase)
                elif nvalue == 'specdiff'      :
                         val.append(aubio_onset_specdiff)
                elif nvalue == 'energy'        :
                         val.append(aubio_onset_energy)
                elif nvalue == 'kl'            :
                         val.append(aubio_onset_kl)
                elif nvalue == 'mkl'           :
                         val.append(aubio_onset_mkl)
                elif nvalue == 'dual'          :
                         val.append('dual')
                else:
                         import sys
                         print "unknown onset detection function selected"
                         sys.exit(1)
                setattr(parser.values, option.dest, val)

def check_pitch_mode(option, opt, value, parser):
        """ utility function to convert a string to aubio_pitchdetection_type"""
        nvalues = parser.rargs[0].split(',')
        val = []
        for nvalue in nvalues:
                if   nvalue == 'mcomb'  :
                         val.append(aubio_pitch_mcomb)
                elif nvalue == 'yin'    :
                         val.append(aubio_pitch_yin)
                elif nvalue == 'fcomb'  :
                         val.append(aubio_pitch_fcomb)
                elif nvalue == 'schmitt':
                         val.append(aubio_pitch_schmitt)
                else:
                         import sys
                         print "error: unknown pitch detection function selected"
                         sys.exit(1)
                setattr(parser.values, option.dest, val)

def check_pitchm_mode(option, opt, value, parser):
        """ utility function to convert a string to aubio_pitchdetection_mode """
        nvalue = parser.rargs[0]
        if   nvalue == 'freq'  :
                 setattr(parser.values, option.dest, aubio_pitchm_freq)
        elif nvalue == 'midi'  :
                 setattr(parser.values, option.dest, aubio_pitchm_midi)
        elif nvalue == 'cent'  :
                 setattr(parser.values, option.dest, aubio_pitchm_cent)
        elif nvalue == 'bin'   :
                 setattr(parser.values, option.dest, aubio_pitchm_bin)
        else:
                 import sys
                 print "error: unknown pitch detection output selected"
                 sys.exit(1)


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
        return mylist, ofunclist

def cutfile(filein,slicetimes,zerothres=0.008,bufsize=1024,hopsize=512):
    frameread = 0
    readsize  = hopsize 
    filei     = sndfile(filein)
    framestep = hopsize/(filei.samplerate()+0.)
    channels  = filei.channels()
    newname   = "%s%s%09.5f%s%s" % (filein.split(".")[0].split("/")[-1],".",
                frameread*framestep,".",filein.split(".")[-1])
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
            fileo = sndfile("%s%s%09.5f%s%s" % 
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
                mylist.append(-1.)
        frameread += 1
    return mylist


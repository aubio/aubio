#! /usr/bin/env python

from aubio import source, tempo
from numpy import median, diff

def get_file_bpm(path, params = {}):
    """ Calculate the beats per minute (bpm) of a given file.
        path: path to the file
        param: dictionary of parameters
    """
    try:
        win_s = params['win_s']
        samplerate = params['samplerate']
        hop_s = params['hop_s']
    except:
        """
        # super fast
        samplerate, win_s, hop_s = 4000, 128, 64 
        # fast
        samplerate, win_s, hop_s = 8000, 512, 128
        """
        # default:
        samplerate, win_s, hop_s = 44100, 1024, 512

    s = source(path, samplerate, hop_s)
    samplerate = s.samplerate
    o = tempo("specdiff", win_s, hop_s, samplerate)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
            #if o.get_confidence() > .2 and len(beats) > 2.:
            #    break
        total_frames += read
        if read < hop_s:
            break

    # Convert to periods and to bpm 
    bpms = 60./diff(beats)
    b = median(bpms)
    return b

if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        bpm = get_file_bpm(f)
        print "%6s" % ("%.2f" % bpm), f

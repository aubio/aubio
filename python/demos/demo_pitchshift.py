#! /usr/bin/env python

import sys
import aubio

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <inputfile> <outputfile> [transpose] [samplerate] [hop_size] [mode]' % sys.argv[0])
        print('available modes: default, crispness:0, crispness:1, ... crispness:6')
        sys.exit(1)
    if len(sys.argv) > 3: transpose = float(sys.argv[3])
    else: transpose = 12.
    if len(sys.argv) > 4: samplerate = int(sys.argv[4])
    else: samplerate = 0
    if len(sys.argv) > 5: hop_size = int(sys.argv[5])
    else: hop_size = 64
    if len(sys.argv) > 6: mode = sys.argv[6]
    else: mode = "default"

    source_read = aubio.source(sys.argv[1], samplerate, hop_size)
    if samplerate == 0: samplerate = source_read.samplerate
    sink_out = aubio.sink(sys.argv[2], samplerate)

    pitchshifter = aubio.pitchshift(mode, 1., hop_size, samplerate)
    if transpose: pitchshifter.set_transpose(transpose)

    total_frames, read = 0, hop_size
    transpose_range = 23.9
    while read:
        vec, read = source_read()
        # transpose the samples
        out = pitchshifter(vec)
        # position in the file (between 0. and 1.)
        percent_read = total_frames / float(source_read.duration)
        # variable transpose rate (in semitones)
        transpose = 2 * transpose_range * percent_read - transpose_range
        # set transpose rate
        pitchshifter.set_transpose(transpose)
        # print the transposition
        #print pitchshifter.get_transpose()
        # write the output
        sink_out(out, read)
        total_frames += read

    # end of file, print some results
    outstr = "wrote %.2fs" % (total_frames / float(samplerate))
    outstr += " (%d frames in" % total_frames
    outstr += " %d blocks" % (total_frames // source_read.hop_size)
    outstr += " at %dHz)" % source_read.samplerate
    outstr += " from " + source_read.uri
    outstr += " to " + sink_out.uri
    print(outstr)

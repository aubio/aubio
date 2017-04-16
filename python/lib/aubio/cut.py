#! /usr/bin/env python

""" this file was written by Paul Brossier
  it is released under the GNU/GPL license.
"""

import sys
import argparse

def parse_args():
    usage = "usage: %s [options] -i soundfile" % sys.argv[0]
    usage += "\n help: %s -h" % sys.argv[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", default=None, nargs='?',
            help="input sound file to analyse", metavar = "<source_file>")
    parser.add_argument("-i", "--input", action = "store", dest = "source_file2",
            help="input sound file to analyse", metavar = "<source_file>")
    parser.add_argument("-O","--onset-method",
            action="store", dest="onset_method", default='default',
            metavar = "<onset_method>",
            help="onset detection method [default=default] \
                    complexdomain|hfc|phase|specdiff|energy|kl|mkl")
    # cutting methods
    parser.add_argument("-b","--beat",
            action="store_true", dest="beat", default=False,
            help="use beat locations")
    """
    parser.add_argument("-S","--silencecut",
            action="store_true", dest="silencecut", default=False,
            help="use silence locations")
    parser.add_argument("-s","--silence",
            metavar = "<value>",
            action="store", dest="silence", default=-70,
            help="silence threshold [default=-70]")
            """
    # algorithm parameters
    parser.add_argument("-r", "--samplerate",
            metavar = "<freq>", type=int,
            action="store", dest="samplerate", default=0,
            help="samplerate at which the file should be represented")
    parser.add_argument("-B","--bufsize",
            action="store", dest="bufsize", default=512,
            metavar = "<size>", type=int,
            help="buffer size [default=512]")
    parser.add_argument("-H","--hopsize",
            metavar = "<size>", type=int,
            action="store", dest="hopsize", default=256,
            help="overlap size [default=256]")
    parser.add_argument("-t","--onset-threshold",
            metavar = "<value>", type=float,
            action="store", dest="threshold", default=0.3,
            help="onset peak picking threshold [default=0.3]")
    parser.add_argument("-c","--cut",
            action="store_true", dest="cut", default=False,
            help="cut input sound file at detected labels \
                    best used with option -L")

    # minioi
    parser.add_argument("-M","--minioi",
            metavar = "<value>", type=str,
            action="store", dest="minioi", default="12ms",
            help="minimum inter onset interval [default=12ms]")

    """
    parser.add_argument("-D","--delay",
            action = "store", dest = "delay", type = float,
            metavar = "<seconds>", default=0,
            help="number of seconds to take back [default=system]\
                    default system delay is 3*hopsize/samplerate")
    parser.add_argument("-C","--dcthreshold",
            metavar = "<value>",
            action="store", dest="dcthreshold", default=1.,
            help="onset peak picking DC component [default=1.]")
    parser.add_argument("-L","--localmin",
            action="store_true", dest="localmin", default=False,
            help="use local minima after peak detection")
    parser.add_argument("-d","--derivate",
            action="store_true", dest="derivate", default=False,
            help="derivate onset detection function")
    parser.add_argument("-z","--zerocross",
            metavar = "<value>",
            action="store", dest="zerothres", default=0.008,
            help="zero-crossing threshold for slicing [default=0.00008]")
            """
    # plotting functions
    """
    parser.add_argument("-p","--plot",
            action="store_true", dest="plot", default=False,
            help="draw plot")
    parser.add_argument("-x","--xsize",
            metavar = "<size>",
            action="store", dest="xsize", default=1.,
            type=float, help="define xsize for plot")
    parser.add_argument("-y","--ysize",
            metavar = "<size>",
            action="store", dest="ysize", default=1.,
            type=float, help="define ysize for plot")
    parser.add_argument("-f","--function",
            action="store_true", dest="func", default=False,
            help="print detection function")
    parser.add_argument("-n","--no-onsets",
            action="store_true", dest="nplot", default=False,
            help="do not plot detected onsets")
    parser.add_argument("-O","--outplot",
            metavar = "<output_image>",
            action="store", dest="outplot", default=None,
            help="save plot to output.{ps,png}")
    parser.add_argument("-F","--spectrogram",
            action="store_true", dest="spectro", default=False,
            help="add spectrogram to the plot")
    """
    parser.add_argument("-o","--output", type = str,
            metavar = "<outputdir>",
            action="store", dest="output_directory", default=None,
            help="specify path where slices of the original file should be created")
    parser.add_argument("--cut-until-nsamples", type = int,
            metavar = "<samples>",
            action = "store", dest = "cut_until_nsamples", default = None,
            help="how many extra samples should be added at the end of each slice")
    parser.add_argument("--cut-every-nslices", type = int,
            metavar = "<samples>",
            action = "store", dest = "cut_every_nslices", default = None,
            help="how many slices should be groupped together at each cut")
    parser.add_argument("--cut-until-nslices", type = int,
            metavar = "<slices>",
            action = "store", dest = "cut_until_nslices", default = None,
            help="how many extra slices should be added at the end of each slice")

    parser.add_argument("-v","--verbose",
            action="store_true", dest="verbose", default=True,
            help="make lots of noise [default]")
    parser.add_argument("-q","--quiet",
            action="store_false", dest="verbose", default=True,
            help="be quiet")
    args = parser.parse_args()
    if not args.source_file and not args.source_file2:
        sys.stderr.write("Error: no file name given\n")
        parser.print_help()
        sys.exit(1)
    elif args.source_file2 is not None:
        args.source_file = args.source_file2
    return args

def main():
    options = parse_args()

    source_file = options.source_file
    hopsize = options.hopsize
    bufsize = options.bufsize
    samplerate = options.samplerate
    source_file = options.source_file

    from aubio import onset, tempo, source

    s = source(source_file, samplerate, hopsize)
    if samplerate == 0: samplerate = s.get_samplerate()

    if options.beat:
        o = tempo(options.onset_method, bufsize, hopsize, samplerate=samplerate)
    else:
        o = onset(options.onset_method, bufsize, hopsize, samplerate=samplerate)
        if options.minioi:
            if options.minioi.endswith('ms'):
                o.set_minioi_ms(int(options.minioi[:-2]))
            elif options.minioi.endswith('s'):
                o.set_minioi_s(int(options.minioi[:-1]))
            else:
                o.set_minioi(int(options.minioi))
    o.set_threshold(options.threshold)

    timestamps = []
    total_frames = 0
    # analyze pass
    while True:
        samples, read = s()
        if o(samples):
            timestamps.append (o.get_last())
            if options.verbose: print ("%.4f" % o.get_last_s())
        total_frames += read
        if read < hopsize: break
    del s
    # print some info
    nstamps = len(timestamps)
    duration = float (total_frames) / float(samplerate)
    info = 'found %(nstamps)d timestamps in %(source_file)s' % locals()
    info += ' (total %(duration).2fs at %(samplerate)dHz)\n' % locals()
    sys.stderr.write(info)

    # cutting pass
    if options.cut and nstamps > 0:
        # generate output files
        from aubio.slicing import slice_source_at_stamps
        timestamps_end = None
        if options.cut_every_nslices:
            timestamps = timestamps[::options.cut_every_nslices]
            nstamps = len(timestamps)
        if options.cut_until_nslices and options.cut_until_nsamples:
            print ("warning: using cut_until_nslices, but cut_until_nsamples is set")
        if options.cut_until_nsamples:
            timestamps_end = [t + options.cut_until_nsamples for t in timestamps[1:]]
            timestamps_end += [ 1e120 ]
        if options.cut_until_nslices:
            timestamps_end = [t for t in timestamps[1 + options.cut_until_nslices:]]
            timestamps_end += [ 1e120 ] * (options.cut_until_nslices + 1)
        slice_source_at_stamps(source_file, timestamps, timestamps_end = timestamps_end,
                output_dir = options.output_directory,
                samplerate = samplerate)

        # print some info
        duration = float (total_frames) / float(samplerate)
        info = 'created %(nstamps)d slices from %(source_file)s' % locals()
        info += ' (total %(duration).2fs at %(samplerate)dHz)\n' % locals()
        sys.stderr.write(info)

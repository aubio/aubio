#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""aubio command line tool

This file was written by Paul Brossier <piem@aubio.org> and is released under
the GNU/GPL v3.

Note: this script is mostly about parsing command line arguments. For more
readable code examples, check out the `python/demos` folder."""

import sys
import argparse
import aubio

def aubio_parser():
    epilog = 'use "%(prog)s <command> --help" for more info about each command'
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('-V', '--version', help="show version",
            action="store_true", dest="show_version")

    subparsers = parser.add_subparsers(title='commands', dest='command',
            metavar="")

    # onset subcommand
    subparser = subparsers.add_parser('onset',
            help='estimate time of onsets (beginning of sound event)',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser)
    helpstr = "onset novelty function"
    helpstr += " <default|energy|hfc|complex|phase|specdiff|kl|mkl|specflux>"
    parser_add_method(subparser, helpstr=helpstr)
    parser_add_threshold(subparser)
    parser_add_silence(subparser)
    parser_add_minioi(subparser)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_onset)

    # pitch subcommand
    subparser = subparsers.add_parser('pitch',
            help='estimate fundamental frequency (monophonic)')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser, buf_size=2048)
    helpstr = "pitch detection method <default|yinfft|yin|mcomb|fcomb|schmitt>"
    parser_add_method(subparser, helpstr=helpstr)
    parser_add_threshold(subparser)
    parser_add_pitch_unit(subparser)
    parser_add_silence(subparser)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_pitch)

    # beat subcommand
    subparser = subparsers.add_parser('beat',
            help='estimate location of beats')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser, buf_size=1024, hop_size=512)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_beat)

    # tempo subcommand
    subparser = subparsers.add_parser('tempo',
            help='estimate overall tempo in bpm')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser, buf_size=1024, hop_size=512)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_tempo)

    # notes subcommand
    subparser = subparsers.add_parser('notes',
            help='estimate midi-like notes (monophonic)')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_notes)

    # mfcc subcommand
    subparser = subparsers.add_parser('mfcc',
            help='extract Mel-Frequency Cepstrum Coefficients')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_mfcc)

    # melbands subcommand
    subparser = subparsers.add_parser('melbands',
            help='extract energies in Mel-frequency bands')
    parser_add_input(subparser)
    parser_add_buf_hop_size(subparser)
    parser_add_time_format(subparser)
    parser_add_verbose_help(subparser)
    subparser.set_defaults(process=process_melbands)

    return parser

def parser_add_input(parser):
    parser.add_argument("source_uri", default=None, nargs='?',
            help="input sound file to analyse", metavar = "<source_uri>")
    parser.add_argument("-i", "--input", dest = "source_uri2",
            help="input sound file to analyse", metavar = "<source_uri>")
    parser.add_argument("-r", "--samplerate",
            metavar = "<freq>", type=int,
            action="store", dest="samplerate", default=0,
            help="samplerate at which the file should be represented")

def parser_add_verbose_help(parser):
    parser.add_argument("-v","--verbose",
            action="count", dest="verbose", default=1,
            help="make lots of noise [default]")
    parser.add_argument("-q","--quiet",
            action="store_const", dest="verbose", const=0,
            help="be quiet")

def parser_add_buf_hop_size(parser, buf_size=512, hop_size=256):
    parser.add_argument("-B","--bufsize",
            action="store", dest="buf_size", default=buf_size,
            metavar = "<size>", type=int,
            help="buffer size [default=%d]" % buf_size)
    parser.add_argument("-H","--hopsize",
            metavar = "<size>", type=int,
            action="store", dest="hop_size", default=hop_size,
            help="overlap size [default=%d]" % hop_size)

def parser_add_method(parser, method='default', helpstr='method'):
    parser.add_argument("-m","--method",
            metavar = "<method>", type=str,
            action="store", dest="method", default=method,
            help="%s [default=%s]" % (helpstr, method))

def parser_add_threshold(parser, default=None):
    parser.add_argument("-t","--threshold",
            metavar = "<threshold>", type=float,
            action="store", dest="threshold", default=default,
            help="threshold [default=%s]" % default)

def parser_add_silence(parser):
    parser.add_argument("-s", "--silence",
            metavar = "<value>", type=float,
            action="store", dest="silence", default=-70,
            help="silence threshold")

def parser_add_minioi(parser):
    parser.add_argument("-M", "--minioi",
            metavar = "<value>", type=str,
            action="store", dest="minioi", default="12ms",
            help="minimum Inter-Onset Interval")

def parser_add_pitch_unit(parser, default="Hz"):
    help_str = "frequency unit, should be one of Hz, midi, bin, cent"
    help_str += " [default=%s]" % default
    parser.add_argument("-u", "--pitch-unit",
            metavar = "<value>", type=str,
            action="store", dest="pitch_unit", default=default,
            help=help_str)

def parser_add_time_format(parser):
    helpstr = "select time values output format (samples, ms, seconds)"
    helpstr += " [default=seconds]"
    parser.add_argument("-T", "--time-format",
             metavar='format',
             dest="time_format",
             default=None,
             help=helpstr)

# some utilities

def samples2seconds(n_frames, samplerate):
    return "%f\t" % (n_frames / float(samplerate))

def samples2milliseconds(n_frames, samplerate):
    return "%f\t" % (1000. * n_frames / float(samplerate))

def samples2samples(n_frames, samplerate):
    return "%d\t" % n_frames

def timefunc(mode):
    if mode is None or mode == 'seconds' or mode == 's':
        return samples2seconds
    elif mode == 'ms' or mode == 'milliseconds':
        return samples2milliseconds
    elif mode == 'samples':
        return samples2samples
    else:
        raise ValueError("invalid time format '%s'" % mode)

# definition of processing classes

class default_process(object):
    def __init__(self, args):
        if 'time_format' in args:
            self.time2string = timefunc(args.time_format)
        if args.verbose > 2 and hasattr(self, 'options'):
            name = type(self).__name__.split('_')[1]
            optstr = ' '.join(['running', name, 'with options', repr(self.options), '\n'])
            sys.stderr.write(optstr)
    def flush(self, n_frames, samplerate):
        # optionally called at the end of process
        pass

    def parse_options(self, args, valid_opts):
        # get any valid options found in a dictionnary of arguments
        options = {k :v for k,v in vars(args).items() if k in valid_opts}
        self.options = options

    def remap_pvoc_options(self, options):
        # FIXME: we need to remap buf_size to win_s, hop_size to hop_s
        # adjust python/ext/py-phasevoc.c to understand buf_size/hop_size
        if 'buf_size' in options:
            options['win_s'] = options['buf_size']
            del options['buf_size']
        if 'hop_size' in options:
            options['hop_s'] = options['hop_size']
            del options['hop_size']
        self.options = options

class process_onset(default_process):
    valid_opts = ['method', 'hop_size', 'buf_size', 'samplerate']
    def __init__(self, args):
        self.parse_options(args, self.valid_opts)
        self.onset = aubio.onset(**self.options)
        if args.threshold is not None:
            self.onset.set_threshold(args.threshold)
        if args.minioi:
            if args.minioi.endswith('ms'):
                self.onset.set_minioi_ms(float(args.minioi[:-2]))
            elif args.minioi.endswith('s'):
                self.onset.set_minioi_s(float(args.minioi[:-1]))
            else:
                self.onset.set_minioi(int(args.minioi))
        if args.silence:
            self.onset.set_silence(args.silence)
        super(process_onset, self).__init__(args)
    def __call__(self, block):
        return self.onset(block)
    def repr_res(self, res, frames_read, samplerate):
        if res[0] != 0:
            outstr = self.time2string(self.onset.get_last(), samplerate)
            sys.stdout.write(outstr + '\n')

class process_pitch(default_process):
    valid_opts = ['method', 'hop_size', 'buf_size', 'samplerate']
    def __init__(self, args):
        self.parse_options(args, self.valid_opts)
        self.pitch = aubio.pitch(**self.options)
        if args.pitch_unit is not None:
            self.pitch.set_unit(args.pitch_unit)
        if args.threshold is not None:
            self.pitch.set_tolerance(args.threshold)
        if args.silence is not None:
            self.pitch.set_silence(args.silence)
        super(process_pitch, self).__init__(args)
    def __call__(self, block):
        return self.pitch(block)
    def repr_res(self, res, frames_read, samplerate):
        fmt_out = self.time2string(frames_read, samplerate)
        sys.stdout.write(fmt_out + "%.6f\n" % res[0])

class process_beat(default_process):
    valid_opts = ['method', 'hop_size', 'buf_size', 'samplerate']
    def __init__(self, args):
        self.parse_options(args, self.valid_opts)
        self.tempo = aubio.tempo(**self.options)
        super(process_beat, self).__init__(args)
    def __call__(self, block):
        return self.tempo(block)
    def repr_res(self, res, frames_read, samplerate):
        if res[0] != 0:
            outstr = self.time2string(self.tempo.get_last(), samplerate)
            sys.stdout.write(outstr + '\n')

class process_tempo(process_beat):
    def __init__(self, args):
        super(process_tempo, self).__init__(args)
        self.beat_locations = []
    def repr_res(self, res, frames_read, samplerate):
        if res[0] != 0:
            self.beat_locations.append(self.tempo.get_last_s())
    def flush(self, frames_read, samplerate):
        import numpy as np
        if len(self.beat_locations) < 2:
            outstr = "unknown bpm"
        else:
            bpms = 60./ np.diff(self.beat_locations)
            median_bpm = np.mean(bpms)
            if len(self.beat_locations) < 10:
                outstr = "%.2f bpm (uncertain)" % median_bpm
            else:
                outstr = "%.2f bpm" % median_bpm
        sys.stdout.write(outstr + '\n')

class process_notes(default_process):
    valid_opts = ['method', 'hop_size', 'buf_size', 'samplerate']
    def __init__(self, args):
        self.parse_options(args, self.valid_opts)
        self.notes = aubio.notes(**self.options)
        super(process_notes, self).__init__(args)
    def __call__(self, block):
        return self.notes(block)
    def repr_res(self, res, frames_read, samplerate):
        if res[2] != 0: # note off
            fmt_out = self.time2string(frames_read, samplerate)
            sys.stdout.write(fmt_out + '\n')
        if res[0] != 0: # note on
            lastmidi = res[0]
            fmt_out = "%f\t" % lastmidi
            fmt_out += self.time2string(frames_read, samplerate)
            sys.stdout.write(fmt_out) # + '\t')
    def flush(self, frames_read, samplerate):
        eof = self.time2string(frames_read, samplerate)
        sys.stdout.write(eof + '\n')

class process_mfcc(default_process):
    def __init__(self, args):
        valid_opts1 = ['hop_size', 'buf_size']
        self.parse_options(args, valid_opts1)
        self.remap_pvoc_options(self.options)
        self.pv = aubio.pvoc(**self.options)

        valid_opts2 = ['buf_size', 'n_filters', 'n_coeffs', 'samplerate']
        self.parse_options(args, valid_opts2)
        self.mfcc = aubio.mfcc(**self.options)

        # remember all options
        self.parse_options(args, list(set(valid_opts1 + valid_opts2)))

        super(process_mfcc, self).__init__(args)

    def __call__(self, block):
        fftgrain = self.pv(block)
        return self.mfcc(fftgrain)
    def repr_res(self, res, frames_read, samplerate):
        fmt_out = self.time2string(frames_read, samplerate)
        fmt_out += ' '.join(["% 9.7f" % f for f in res.tolist()])
        sys.stdout.write(fmt_out + '\n')

class process_melbands(default_process):
    def __init__(self, args):
        self.args = args
        valid_opts = ['hop_size', 'buf_size']
        self.parse_options(args, valid_opts)
        self.remap_pvoc_options(self.options)
        self.pv = aubio.pvoc(**self.options)

        valid_opts = ['buf_size', 'n_filters']
        self.parse_options(args, valid_opts)
        self.remap_pvoc_options(self.options)
        self.filterbank = aubio.filterbank(**self.options)
        self.filterbank.set_mel_coeffs_slaney(args.samplerate)

        super(process_melbands, self).__init__(args)
    def __call__(self, block):
        fftgrain = self.pv(block)
        return self.filterbank(fftgrain)
    def repr_res(self, res, frames_read, samplerate):
        fmt_out = self.time2string(frames_read, samplerate)
        fmt_out += ' '.join(["% 9.7f" % f for f in res.tolist()])
        sys.stdout.write(fmt_out + '\n')

def main():
    parser = aubio_parser()
    args = parser.parse_args()
    if 'show_version' in args and args.show_version:
        sys.stdout.write('aubio version ' + aubio.version + '\n')
        sys.exit(0)
    elif 'verbose' in args and args.verbose > 3:
        sys.stderr.write('aubio version ' + aubio.version + '\n')
    if 'command' not in args or args.command is None:
        # no command given, print help and return 1
        parser.print_help()
        sys.exit(1)
    elif not args.source_uri and not args.source_uri2:
        sys.stderr.write("Error: a source is required\n")
        parser.print_help()
        sys.exit(1)
    elif args.source_uri2 is not None:
        args.source_uri = args.source_uri2
    try:
        # open source_uri
        with aubio.source(args.source_uri, hop_size=args.hop_size,
                samplerate=args.samplerate) as a_source:
            # always update args.samplerate to native samplerate, in case
            # source was opened with args.samplerate=0
            args.samplerate = a_source.samplerate
            # create the processor for this subcommand
            processor = args.process(args)
            frames_read = 0
            while True:
                # read new block from source
                block, read = a_source()
                # execute processor on this block
                res = processor(block)
                # print results for this block
                if args.verbose > 0:
                    processor.repr_res(res, frames_read, a_source.samplerate)
                # increment total number of frames read
                frames_read += read
                # exit loop at end of file
                if read < a_source.hop_size: break
            # flush the processor if needed
            processor.flush(frames_read, a_source.samplerate)
            if args.verbose > 1:
                fmt_string = "read {:.2f}s"
                fmt_string += " ({:d} samples in {:d} blocks of {:d})"
                fmt_string += " from {:s} at {:d}Hz\n"
                sys.stderr.write(fmt_string.format(
                        frames_read/float(a_source.samplerate),
                        frames_read,
                        frames_read // a_source.hop_size + 1,
                        a_source.hop_size,
                        a_source.uri,
                        a_source.samplerate))
    except KeyboardInterrupt:
        sys.exit(1)

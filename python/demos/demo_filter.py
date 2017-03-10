#! /usr/bin/env python


def apply_filter(path):
    from aubio import source, sink, digital_filter
    from os.path import basename, splitext

    # open input file, get its samplerate
    s = source(path)
    samplerate = s.samplerate

    # create an A-weighting filter
    f = digital_filter(7)
    f.set_a_weighting(samplerate)
    # alternatively, apply another filter

    # create output file
    o = sink("filtered_" + splitext(basename(path))[0] + ".wav", samplerate)

    total_frames = 0
    while True:
        samples, read = s()
        filtered_samples = f(samples)
        o(filtered_samples, read)
        total_frames += read
        if read < s.hop_size: break

    duration = total_frames / float(samplerate)
    print ("read {:s}".format(s.uri))
    print ("applied A-weighting filtered ({:d} Hz)".format(samplerate))
    print ("wrote {:s} ({:.2f} s)".format(o.uri, duration))

if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        apply_filter(f)

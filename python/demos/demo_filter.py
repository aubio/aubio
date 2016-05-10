#! /usr/bin/env python


def apply_filter(path, params = {}):
    from aubio import source, sink, digital_filter
    from os.path import basename, splitex, splitextt
    s = source(path)
    f = digital_filter(7)
    f.set_a_weighting(s.samplerate)
    #f = digital_filter(3)
    #f.set_biquad(...)
    o = sink("filtered_" + splitext(basename(path))[0] + ".wav")
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = s()
        filtered_samples = f(samples)
        o(samples, read)
        total_frames += read
        if read < s.hop_size: break
    print "filtered", s.uri, "to", o.uri, "using an A-weighting filter"

if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        apply_filter(f)

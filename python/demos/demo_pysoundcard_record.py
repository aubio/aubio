#! /usr/bin/env python

def record_sink(sink_path):
    """Record an audio file using pysoundcard."""

    from aubio import sink
    from pysoundcard import Stream

    hop_size = 256
    duration = 5 # in seconds
    s = Stream(block_length = hop_size)
    g = sink(sink_path, samplerate = s.sample_rate)

    s.start()
    total_frames = 0
    while total_frames < duration * s.sample_rate:
        vec = s.read(hop_size)
        # mix down to mono
        mono_vec = vec.sum(-1) / float(s.input_channels)
        g(mono_vec, hop_size)
        total_frames += hop_size
    s.stop()

if __name__ == '__main__':
    import sys
    record_sink(sys.argv[1])

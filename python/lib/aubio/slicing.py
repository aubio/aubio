from aubio import source, sink
import os

def slice_source_at_stamps(source_file, timestamps, timestamps_end = None,
        output_dir = None,
        samplerate = 0,
        hopsize = 256):

    if timestamps == None or len(timestamps) == 0:
        raise ValueError ("no timestamps given")

    if timestamps_end != None and len(timestamps_end) != len(timestamps):
        raise ValueError ("len(timestamps_end) != len(timestamps)")

    source_base_name, source_ext = os.path.splitext(os.path.basename(source_file))
    if output_dir != None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        source_base_name = os.path.join(output_dir, source_base_name)

    def new_sink_name(source_base_name, timestamp):
        return source_base_name + '_%02.3f' % (timestamp) + '.wav'

    # reopen source file
    s = source(source_file, samplerate, hopsize)
    if samplerate == 0: samplerate = s.get_samplerate()
    # create first sink at 0
    g = sink(new_sink_name(source_base_name, 0.), samplerate)
    total_frames = 0
    # get next region
    next_stamp = int(timestamps.pop(0))

    while True:
        # get hopsize new samples from source
        vec, read = s()
        remaining = next_stamp - total_frames
        # not enough frames remaining, time to split
        if remaining < read:
            if remaining != 0:
                # write remaining samples from current region
                g(vec[0:remaining], remaining)
            # close this file
            del g
            # create a new file for the new region
            new_sink_path = new_sink_name(source_base_name, next_stamp / float(samplerate))
            #print "new slice", total_frames, "+", remaining, "=", next_stamp
            g = sink(new_sink_path, samplerate)
            # write the remaining samples in the new file
            g(vec[remaining:read], read - remaining)
            if len(timestamps):
                next_stamp = int(timestamps.pop(0))
            else:
                next_stamp = 1e120
        else:
            g(vec[0:read], read)
        total_frames += read
        if read < hopsize: break

    # close the last file
    del g

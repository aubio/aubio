from aubio import source, sink
import os

max_timestamp = 1e120

def slice_source_at_stamps(source_file, timestamps, timestamps_end = None,
        output_dir = None,
        samplerate = 0,
        hopsize = 256):

    if timestamps == None or len(timestamps) == 0:
        raise ValueError ("no timestamps given")

    if timestamps[0] != 0:
        timestamps = [0] + timestamps

    if timestamps_end != None and len(timestamps_end) != len(timestamps):
        raise ValueError ("len(timestamps_end) != len(timestamps)")
    else:
        timestamps_end = [t - 1 for t in timestamps[1:] ] + [ max_timestamp ]

    source_base_name, source_ext = os.path.splitext(os.path.basename(source_file))
    if output_dir != None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        source_base_name = os.path.join(output_dir, source_base_name)

    def new_sink_name(source_base_name, timestamp, samplerate):
        timestamp_seconds = timestamp / float(samplerate)
        #print source_base_name + '_%02.3f' % (timestamp_seconds) + '.wav'
        return source_base_name + '_%02.3f' % (timestamp_seconds) + '.wav'

    # reopen source file
    s = source(source_file, samplerate, hopsize)
    if samplerate == 0: samplerate = s.get_samplerate()
    total_frames = 0
    # get next region
    start_stamp = int(timestamps.pop(0))
    end_stamp = int(timestamps_end.pop(0))

    # create first sink
    new_sink_path = new_sink_name(source_base_name, start_stamp, samplerate)
    #print "new slice", total_frames, "+", remaining, "=", end_stamp
    g = sink(new_sink_path, samplerate)

    while True:
        # get hopsize new samples from source
        vec, read = s()
        # number of samples until end of region
        remaining = end_stamp - total_frames
        # not enough frames remaining, time to split
        if remaining < read:
            if remaining != 0:
                # write remaining samples from current region
                g(vec[0:remaining], remaining)
            # close this file
            del g
            # get the next region
            start_stamp = int(timestamps.pop(0))
            end_stamp = int(timestamps_end.pop(0))
            # create a new file for the new region
            new_sink_path = new_sink_name(source_base_name, start_stamp, samplerate)
            #print "new slice", total_frames, "+", remaining, "=", end_stamp
            g = sink(new_sink_path, samplerate)
            # write the remaining samples in the new file
            g(vec[remaining:read], read - remaining)
        elif read > 0:
            # write all the samples
            g(vec[0:read], read)
        total_frames += read
        if read < hopsize: break

    # close the last file
    del g

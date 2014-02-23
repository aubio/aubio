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
        if timestamps_end != None:
            timestamps_end = [timestamps[1] - 1] + timestamps_end

    if timestamps_end != None:
        if len(timestamps_end) != len(timestamps):
            raise ValueError ("len(timestamps_end) != len(timestamps)")
    else:
        timestamps_end = [t - 1 for t in timestamps[1:] ] + [ max_timestamp ]

    regions = zip(timestamps, timestamps_end)
    #print regions

    source_base_name, source_ext = os.path.splitext(os.path.basename(source_file))
    if output_dir != None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        source_base_name = os.path.join(output_dir, source_base_name)

    def new_sink_name(source_base_name, timestamp, samplerate):
        timestamp_seconds = timestamp / float(samplerate)
        return source_base_name + "_%011.6f" % timestamp_seconds + '.wav'

    # reopen source file
    s = source(source_file, samplerate, hopsize)
    samplerate = s.get_samplerate()

    total_frames = 0
    slices = []

    while True:
        # get hopsize new samples from source
        vec, read = s.do_multi()
        # if the total number of frames read will exceed the next region start
        if len(regions) and total_frames + read >= regions[0][0]:
            #print "getting", regions[0], "at", total_frames
            # get next region
            start_stamp, end_stamp = regions.pop(0)
            # create a name for the sink
            new_sink_path = new_sink_name(source_base_name, start_stamp, samplerate)
            # create its sink
            g = sink(new_sink_path, samplerate, s.channels)
            # create a dictionary containing all this
            new_slice = {'start_stamp': start_stamp, 'end_stamp': end_stamp, 'sink': g}
            # append the dictionary to the current list of slices
            slices.append(new_slice)

        for current_slice in slices:
            start_stamp = current_slice['start_stamp']
            end_stamp = current_slice['end_stamp']
            g = current_slice['sink']
            # sample index to start writing from new source vector
            start = max(start_stamp - total_frames, 0)
            # number of samples yet to written be until end of region
            remaining = end_stamp - total_frames + 1
            #print current_slice, remaining, start
            # not enough frames remaining, time to split
            if remaining < read:
                if remaining > start:
                    # write remaining samples from current region
                    g.do_multi(vec[:,start:remaining], remaining - start)
                    #print "closing region", "remaining", remaining
                    # close this file
                    g.close()
            elif read > start:
                # write all the samples
                g.do_multi(vec[:,start:read], read - start)
        total_frames += read
        if read < hopsize: break

#! /usr/bin/env python

def array_from_text_file(filename, dtype = 'float'):
    import os.path
    from numpy import array
    filename = os.path.join(os.path.dirname(__file__), filename)
    return array([line.split() for line in open(filename).readlines()],
        dtype = dtype)

def list_all_sounds(rel_dir):
    import os.path, glob
    datadir = os.path.join(os.path.dirname(__file__), rel_dir)
    return glob.glob(os.path.join(datadir,'*.*'))

def get_default_test_sound(TestCase, rel_dir = 'sounds'):
    all_sounds = list_all_sounds(rel_dir)
    if len(all_sounds) == 0:
        TestCase.skipTest("please add some sounds in \'python/tests/sounds\'")
    else:
        return all_sounds[0]

def get_tmp_sink_path():
    from tempfile import mkstemp
    import os
    fd, path = mkstemp()
    os.close(fd)
    return path

def del_tmp_sink_path(path):
    import os
    os.unlink(path)

def array_from_yaml_file(filename):
    import yaml
    f = open(filename)
    yaml_data = yaml.safe_load(f)
    f.close()
    return yaml_data

def count_samples_in_file(file_path):
    from aubio import source
    hopsize = 256
    s = source(file_path, 0, hopsize)
    total_frames = 0
    while True:
        samples, read = s()
        total_frames += read
        if read < hopsize: break
    return total_frames

def count_samples_in_directory(samples_dir):
    import os
    total_frames = 0
    for f in os.walk(samples_dir):
        if len(f[2]):
            for each in f[2]:
                file_path = os.path.join(f[0], each)
                if file_path:
                    total_frames += count_samples_in_file(file_path)
    return total_frames

def count_files_in_directory(samples_dir):
    import os
    total_files = 0
    for f in os.walk(samples_dir):
        if len(f[2]):
            for each in f[2]:
                file_path = os.path.join(f[0], each)
                if file_path:
                    total_files += 1
    return total_files

import os

header = """// this file is generated! do not modify
#include "aubio-types.h"
"""

skip_objects = [
  # already in ext/
  'fft',
  'pvoc',
  'filter',
  'filterbank',
  #'resampler',
  # AUBIO_UNSTABLE
  'hist',
  'parameter',
  'scale',
  'beattracking',
  'resampler',
  'peakpicker',
  'pitchfcomb',
  'pitchmcomb',
  'pitchschmitt',
  'pitchspecacf',
  'pitchyin',
  'pitchyinfft',
  'sink',
  'sink_apple_audio',
  'sink_sndfile',
  'sink_wavwrite',
  #'mfcc',
  'source',
  'source_apple_audio',
  'source_sndfile',
  'source_avcodec',
  'source_wavread',
  #'sampler',
  'audio_unit',

  'tss',
  ]


def get_cpp_objects():

    cpp_output = [l.strip() for l in os.popen('cpp -DAUBIO_UNSTABLE=1 -I../build/src ../src/aubio.h').readlines()]
    #cpp_output = [l.strip() for l in os.popen('cpp -DAUBIO_UNSTABLE=0 -I../build/src ../src/onset/onset.h').readlines()]
    #cpp_output = [l.strip() for l in os.popen('cpp -DAUBIO_UNSTABLE=0 -I../build/src ../src/pitch/pitch.h').readlines()]

    cpp_output = filter(lambda y: len(y) > 1, cpp_output)
    cpp_output = filter(lambda y: not y.startswith('#'), cpp_output)
    cpp_output = list(cpp_output)

    i = 1
    while 1:
        if i >= len(cpp_output): break
        if cpp_output[i-1].endswith(',') or cpp_output[i-1].endswith('{') or cpp_output[i].startswith('}'):
            cpp_output[i] = cpp_output[i-1] + ' ' + cpp_output[i]
            cpp_output.pop(i-1)
        else:
            i += 1

    typedefs = filter(lambda y: y.startswith ('typedef struct _aubio'), cpp_output)

    cpp_objects = [a.split()[3][:-1] for a in typedefs]

    return cpp_output, cpp_objects

def generate_external(output_path):
    os.mkdir(output_path)
    sources_list = []
    cpp_output, cpp_objects = get_cpp_objects()
    lib = {}

    for o in cpp_objects:
        if o[:6] != 'aubio_':
            continue
        shortname = o[6:-2]
        if shortname in skip_objects:
            continue
        lib[shortname] = {'struct': [], 'new': [], 'del': [], 'do': [], 'get': [], 'set': [], 'other': []}
        lib[shortname]['longname'] = o
        lib[shortname]['shortname'] = shortname
        for fn in cpp_output:
            if o[:-1] in fn:
                #print "found", o[:-1], "in", fn
                if 'typedef struct ' in fn:
                    lib[shortname]['struct'].append(fn)
                elif '_do' in fn:
                    lib[shortname]['do'].append(fn)
                elif 'new_' in fn:
                    lib[shortname]['new'].append(fn)
                elif 'del_' in fn:
                    lib[shortname]['del'].append(fn)
                elif '_get_' in fn:
                    lib[shortname]['get'].append(fn)
                elif '_set_' in fn:
                    lib[shortname]['set'].append(fn)
                else:
                    #print "no idea what to do about", fn
                    lib[shortname]['other'].append(fn)

    """
    for fn in cpp_output:
        found = 0
        for o in lib:
            for family in lib[o]:
                if fn in lib[o][family]:
                    found = 1
        if found == 0:
            print "missing", fn

    for o in lib:
        for family in lib[o]:
            if type(lib[o][family]) == str:
                print ( "{:15s} {:10s} {:s}".format(o, family, lib[o][family] ) )
            elif len(lib[o][family]) == 1:
                print ( "{:15s} {:10s} {:s}".format(o, family, lib[o][family][0] ) )
            else:                                             
                print ( "{:15s} {:10s} {:d}".format(o, family, len(lib[o][family]) ) )
    """

    from .gen_code import MappedObject
    for o in lib:
        out = header
        mapped = MappedObject(lib[o])
        out += mapped.gen_code()
        output_file = os.path.join(output_path, 'gen-%s.c' % o)
        with open(output_file, 'w') as f:
            f.write(out)
            print ("wrote %s" % output_file )
            sources_list.append(output_file)

    out = header
    out += "#include \"aubio-generated.h\""
    check_types = "\n     ||  ".join(["PyType_Ready(&Py_%sType) < 0" % o for o in lib])
    out += """

int generated_types_ready (void)
{{
  return ({pycheck_types});
}}
""".format(pycheck_types = check_types)

    add_types = "".join(["""
  Py_INCREF (&Py_{name}Type);
  PyModule_AddObject(m, "{name}", (PyObject *) & Py_{name}Type);""".format(name = o) for o in lib])
    out += """

void add_generated_objects ( PyObject *m )
{{
{add_types}
}}
""".format(add_types = add_types)

    output_file = os.path.join(output_path, 'aubio-generated.c')
    with open(output_file, 'w') as f:
        f.write(out)
        print ("wrote %s" % output_file )
        sources_list.append(output_file)
    
    objlist = "".join(["extern PyTypeObject Py_%sType;\n" % p for p in lib])
    out = """
// generated list of objects created with gen_external.py
#include <Python.h>

{objlist}
int generated_objects ( void );
void add_generated_objects( PyObject *m );
""".format(objlist = objlist)

    output_file = os.path.join(output_path, 'aubio-generated.h')
    with open(output_file, 'w') as f:
        f.write(out)
        print ("wrote %s" % output_file )
        # no need to add header to list of sources

    return sources_list

if __name__ == '__main__':
    output_path = 'gen'
    generate_external(output_path)

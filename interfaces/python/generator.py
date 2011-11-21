#! /usr/bin/python

""" This file generates a c file from a list of cpp prototypes. """

import os, sys

skip_objects = ['fft', 'pvoc', 'filter', 'filterbank', 'resampler']

cpp_output = [l.strip() for l in os.popen('cpp -DAUBIO_UNSTABLE=1 -I../../build/src ../../src/aubio.h').readlines()]

cpp_output = filter(lambda y: len(y) > 1, cpp_output)
cpp_output = filter(lambda y: not y.startswith('#'), cpp_output)

i = 1
while 1:
    if i >= len(cpp_output): break
    if cpp_output[i-1].endswith(',') or cpp_output[i-1].endswith('{') or cpp_output[i].startswith('}'):
        cpp_output[i] = cpp_output[i-1] + ' ' + cpp_output[i]
        cpp_output.pop(i-1)
    else:
        i += 1

typedefs = filter(lambda y: y.startswith ('typedef struct _aubio'), cpp_output)

objects = [a.split()[3][:-1] for a in typedefs]

print "-- INFO: %d objects in total" % len(objects)

generated_objects = []

for this_object in objects:
    lint = 0
 
    if this_object[-2:] == '_t':
        object_name = this_object[:-2]
    else:
        object_name = this_object
        print "-- WARNING: %s does not end in _t" % this_object

    if object_name[:len('aubio_')] != 'aubio_':
        print "-- WARNING: %s does not start n aubio_" % this_object

    print "-- INFO: looking at", object_name
    object_methods = filter(lambda x: this_object in x, cpp_output)
    object_methods = [a.strip() for a in object_methods]
    object_methods = filter(lambda x: not x.startswith('typedef'), object_methods)
    #for method in object_methods:
    #    print method

    new_methods = filter(lambda x: 'new_'+object_name in x, object_methods)
    if len(new_methods) > 1:
        print "-- WARNING: more than one new method for", object_name
        for method in new_methods:
            print method
    elif len(new_methods) < 1:
        print "-- WARNING: no new method for", object_name
    elif 0:
        for method in new_methods:
            print method

    del_methods = filter(lambda x: 'del_'+object_name in x, object_methods)
    if len(del_methods) > 1:
        print "-- WARNING: more than one del method for", object_name
        for method in del_methods:
            print method
    elif len(del_methods) < 1:
        print "-- WARNING: no del method for", object_name

    do_methods = filter(lambda x: object_name+'_do' in x, object_methods)
    if len(do_methods) > 1:
        pass
        #print "-- WARNING: more than one do method for", object_name
        #for method in do_methods:
        #    print method
    elif len(do_methods) < 1:
        print "-- WARNING: no do method for", object_name
    elif 0:
        for method in do_methods:
            print method

    # check do methods return void
    for method in do_methods:
        if (method.split()[0] != 'void'):
            print "-- ERROR: _do method does not return void:", method 

    get_methods = filter(lambda x: object_name+'_get_' in x, object_methods)

    set_methods = filter(lambda x: object_name+'_set_' in x, object_methods)
    for method in set_methods:
        if (method.split()[0] != 'uint_t'):
            print "-- ERROR: _set method does not return uint_t:", method 

    other_methods = filter(lambda x: x not in new_methods, object_methods)
    other_methods = filter(lambda x: x not in del_methods, other_methods)
    other_methods = filter(lambda x: x not in    do_methods, other_methods)
    other_methods = filter(lambda x: x not in get_methods, other_methods)
    other_methods = filter(lambda x: x not in set_methods, other_methods)

    if len(other_methods) > 0:
        print "-- WARNING: some methods for", object_name, "were unidentified"
        for method in other_methods:
            print method

    # generate this_object
    if not os.path.isdir('generated'): os.mkdir('generated')
    from gen_pyobject import *
    short_name = object_name[len('aubio_'):]
    if short_name in skip_objects:
            print "-- INFO: skipping object", short_name 
            continue
    if 1: #try:
        s = gen_new_init(new_methods[0], short_name)
        s += gen_do(do_methods[0], short_name) 
        s += gen_members(new_methods[0], short_name)
        s += gen_methods(get_methods, set_methods, short_name)
        s += gen_finish(short_name)
        fd = open('generated/gen-'+short_name+'.c', 'w')
        fd.write(s)
    #except Exception, e:
    #        print "-- ERROR:", type(e), str(e), "in", short_name
    #        continue
    generated_objects += [this_object]


s = """// generated list of generated objects

"""

for each in generated_objects:
    s += "extern PyTypeObject Py_%sType;\n" % \
            each.replace('aubio_','').replace('_t','')

types_ready = []
for each in generated_objects:
    types_ready.append("  PyType_Ready (&Py_%sType) < 0" % \
            each.replace('aubio_','').replace('_t','') )

s += """
int
generated_types_ready (void)
{
    return ("""
s += ('||\n').join(types_ready)
s += """);
}
"""

s += """
void
add_generated_objects ( PyObject *m )
{"""
for each in generated_objects:
    s += """  Py_INCREF (&Py_%(name)sType);
  PyModule_AddObject (m, "%(name)s", (PyObject *) & Py_%(name)sType);""" % \
          { 'name': ( each.replace('aubio_','').replace('_t','') ) }

s += """
}"""

fd = open('generated/aubio-generated.h', 'w')
fd.write(s)

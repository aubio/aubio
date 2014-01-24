#! /usr/bin/python

""" This file generates a c file from a list of cpp prototypes. """

import os, sys, shutil
from gen_pyobject import write_msg, gen_new_init, gen_do, gen_members, gen_methods, gen_finish

def get_cpp_objects():

  cpp_output = [l.strip() for l in os.popen('cpp -DAUBIO_UNSTABLE=1 -I../build/src ../src/aubio.h').readlines()]

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

  cpp_objects = [a.split()[3][:-1] for a in typedefs]

  return cpp_output, cpp_objects

def generate_object_files(output_path):
  if os.path.isdir(output_path): shutil.rmtree(output_path)
  os.mkdir(output_path)

  generated_objects = []
  cpp_output, cpp_objects = get_cpp_objects()
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
      'sndfile',
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
      'source',
      'source_apple_audio',
      'source_sndfile',
      'source_avcodec',
      'source_wavread',
      #'sampler',
      'audio_unit',
      ]

  write_msg("-- INFO: %d objects in total" % len(cpp_objects))

  for this_object in cpp_objects:
      lint = 0

      if this_object[-2:] == '_t':
          object_name = this_object[:-2]
      else:
          object_name = this_object
          write_msg("-- WARNING: %s does not end in _t" % this_object)

      if object_name[:len('aubio_')] != 'aubio_':
          write_msg("-- WARNING: %s does not start n aubio_" % this_object)

      write_msg("-- INFO: looking at", object_name)
      object_methods = filter(lambda x: this_object in x, cpp_output)
      object_methods = [a.strip() for a in object_methods]
      object_methods = filter(lambda x: not x.startswith('typedef'), object_methods)
      #for method in object_methods:
      #    write_msg(method)
      new_methods = filter(lambda x: 'new_'+object_name in x, object_methods)
      if len(new_methods) > 1:
          write_msg("-- WARNING: more than one new method for", object_name)
          for method in new_methods:
              write_msg(method)
      elif len(new_methods) < 1:
          write_msg("-- WARNING: no new method for", object_name)
      elif 0:
          for method in new_methods:
              write_msg(method)

      del_methods = filter(lambda x: 'del_'+object_name in x, object_methods)
      if len(del_methods) > 1:
          write_msg("-- WARNING: more than one del method for", object_name)
          for method in del_methods:
              write_msg(method)
      elif len(del_methods) < 1:
          write_msg("-- WARNING: no del method for", object_name)

      do_methods = filter(lambda x: object_name+'_do' in x, object_methods)
      if len(do_methods) > 1:
          pass
          #write_msg("-- WARNING: more than one do method for", object_name)
          #for method in do_methods:
          #    write_msg(method)
      elif len(do_methods) < 1:
          write_msg("-- WARNING: no do method for", object_name)
      elif 0:
          for method in do_methods:
              write_msg(method)

      # check do methods return void
      for method in do_methods:
          if (method.split()[0] != 'void'):
              write_msg("-- ERROR: _do method does not return void:", method )

      get_methods = filter(lambda x: object_name+'_get_' in x, object_methods)

      set_methods = filter(lambda x: object_name+'_set_' in x, object_methods)
      for method in set_methods:
          if (method.split()[0] != 'uint_t'):
              write_msg("-- ERROR: _set method does not return uint_t:", method )

      other_methods = filter(lambda x: x not in new_methods, object_methods)
      other_methods = filter(lambda x: x not in del_methods, other_methods)
      other_methods = filter(lambda x: x not in    do_methods, other_methods)
      other_methods = filter(lambda x: x not in get_methods, other_methods)
      other_methods = filter(lambda x: x not in set_methods, other_methods)

      if len(other_methods) > 0:
          write_msg("-- WARNING: some methods for", object_name, "were unidentified")
          for method in other_methods:
              write_msg(method)


      # generate this_object
      short_name = object_name[len('aubio_'):]
      if short_name in skip_objects:
              write_msg("-- INFO: skipping object", short_name )
              continue
      if 1: #try:
          s = gen_new_init(new_methods[0], short_name)
          s += gen_do(do_methods[0], short_name)
          s += gen_members(new_methods[0], short_name)
          s += gen_methods(get_methods, set_methods, short_name)
          s += gen_finish(short_name)
          generated_filepath = os.path.join(output_path,'gen-'+short_name+'.c')
          fd = open(generated_filepath, 'w')
          fd.write(s)
      #except Exception, e:
      #        write_msg("-- ERROR:", type(e), str(e), "in", short_name)
      #        continue
      generated_objects += [this_object]

  s = """// generated list of objects created with generator.py

"""

  types_ready = []
  for each in generated_objects:
      types_ready.append("  PyType_Ready (&Py_%sType) < 0" % \
              each.replace('aubio_','').replace('_t','') )

  s = """// generated list of objects created with generator.py

#include "aubio-generated.h"
"""

  s += """
int generated_types_ready (void)
{
  return (
"""
  s += ('\n     ||').join(types_ready)
  s += """);
}
"""

  s += """
void add_generated_objects ( PyObject *m )
{"""
  for each in generated_objects:
    s += """
  Py_INCREF (&Py_%(name)sType);
  PyModule_AddObject (m, "%(name)s", (PyObject *) & Py_%(name)sType);""" % \
          { 'name': ( each.replace('aubio_','').replace('_t','') ) }

  s += """
}"""

  fd = open(os.path.join(output_path,'aubio-generated.c'), 'w')
  fd.write(s)

  s = """// generated list of objects created with generator.py

#include <Python.h>

"""

  for each in generated_objects:
      s += "extern PyTypeObject Py_%sType;\n" % \
              each.replace('aubio_','').replace('_t','')

  s+= "int generated_objects ( void );\n"
  s+= "void add_generated_objects( PyObject *m );\n"

  fd = open(os.path.join(output_path,'aubio-generated.h'), 'w')
  fd.write(s)

  from os import listdir
  generated_files = listdir(output_path)
  generated_files = filter(lambda x: x.endswith('.c'), generated_files)
  generated_files = [output_path+'/'+f for f in generated_files]
  return generated_files

if __name__ == '__main__':
  generate_object_files('gen')

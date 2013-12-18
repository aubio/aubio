#! /usr/bin/python

""" This madness of code is used to generate the C code of the python interface
to aubio. Don't try this at home.

The list of typedefs and functions is obtained from the command line 'cpp
aubio.h'. This list is then used to parse all the functions about this object.

I hear the ones asking "why not use swig, or cython, or something like that?"

The requirements for this extension are the following:

    - aubio vectors can be viewed as numpy arrays, and vice versa
    - aubio 'object' should be python classes, not just a bunch of functions

I haven't met any python interface generator that can meet both these
requirements. If you know of one, please let me know, it will spare me
maintaining this bizarre file.
"""

param_numbers = {
  'source': [0, 2],
  'sink':   [2, 0],
  'sampler': [1, 1],
}

# TODO
# do function: for now, only the following pattern is supported:
# void aubio_<foo>_do (aubio_foo_t * o, 
#       [input1_t * input, [output1_t * output, ..., output3_t * output]]);
# There is no way of knowing that output1 is actually input2. In the future,
# const could be used for the inputs in the C prototypes.

def write_msg(*args):
  pass
  # uncomment out for debugging
  #print args

def split_type(arg):
    """ arg = 'foo *name' 
        return ['foo*', 'name'] """
    l = arg.split()
    type_arg = {'type': l[0], 'name': l[1]}
    # ['foo', '*name'] -> ['foo*', 'name']
    if l[-1].startswith('*'):
        #return [l[0]+'*', l[1][1:]]
        type_arg['type'] = l[0] + '*'
        type_arg['name'] = l[1][1:]
    # ['foo', '*', 'name'] -> ['foo*', 'name']
    if len(l) == 3:
        #return [l[0]+l[1], l[2]]
        type_arg['type'] = l[0]+l[1]
        type_arg['name'] = l[2]
    else:
        #return l
        pass
    return type_arg

def get_params(proto):
    """ get the list of parameters from a function prototype
    example: proto = "int main (int argc, char ** argv)"
    returns: ['int argc', 'char ** argv']
    """
    import re
    paramregex = re.compile('[\(, ](\w+ \*?\*? ?\w+)[, \)]')
    return paramregex.findall(proto)

def get_params_types_names(proto):
    """ get the list of parameters from a function prototype
    example: proto = "int main (int argc, char ** argv)"
    returns: [['int', 'argc'], ['char **','argv']]
    """
    return map(split_type, get_params(proto)) 

def get_return_type(proto):
    import re
    paramregex = re.compile('(\w+ ?\*?).*')
    outputs = paramregex.findall(proto)
    assert len(outputs) == 1
    return outputs[0].replace(' ', '')

def get_name(proto):
    name = proto.split()[1].split('(')[0]
    return name.replace('*','')

# the important bits: the size of the output for each objects. this data should
# move into the C library at some point.
defaultsizes = {
    'resampler':    ['input->length * self->ratio'],
    'specdesc':     ['1'],
    'onset':        ['1'],
    'pitchyin':     ['1'],
    'pitchyinfft':  ['1'],
    'pitchschmitt': ['1'],
    'pitchmcomb':   ['1'],
    'pitchfcomb':   ['1'],
    'pitch':        ['1'],
    'tss':          ['self->buf_size', 'self->buf_size'],
    'mfcc':         ['self->n_coeffs'],
    'beattracking': ['self->hop_size'],
    'tempo':        ['1'],
    'peakpicker':   ['1'],
    'source':       ['self->hop_size', '1'],
    'sampler':      ['self->hop_size'],
    'wavetable':    ['self->hop_size'],
}

# default value for variables
aubioinitvalue = {
    'uint_t': 0,
    'smpl_t': 0,
    'lsmp_t': 0.,
    'char_t*': 'NULL',
    }

aubiodefvalue = {
    # we have some clean up to do
    'buf_size': 'Py_default_vector_length', 
    # and here too
    'hop_size': 'Py_default_vector_length / 2', 
    # these should be alright
    'samplerate': 'Py_aubio_default_samplerate', 
    # now for the non obvious ones
    'n_filters': '40', 
    'n_coeffs': '13', 
    'nelems': '10',
    'flow': '0.', 
    'fhig': '1.', 
    'ilow': '0.', 
    'ihig': '1.', 
    'thrs': '0.5',
    'ratio': '0.5',
    'method': '"default"',
    'uri': '"none"',
    }

# aubio to python
aubio2pytypes = {
    'uint_t': 'I',
    'smpl_t': 'f',
    'lsmp_t': 'd',
    'fvec_t*': 'O',
    'cvec_t*': 'O',
    'char_t*': 's',
}

# python to aubio
aubiovecfrompyobj = {
    'fvec_t*': 'PyAubio_ArrayToCFvec',
    'cvec_t*': 'PyAubio_ArrayToCCvec',
    'uint_t': '(uint_t)PyInt_AsLong',
}

# aubio to python
aubiovectopyobj = {
    'fvec_t*': 'PyAubio_CFvecToArray',
    'cvec_t*': 'PyAubio_CCvecToPyCvec',
    'smpl_t': 'PyFloat_FromDouble',
    'uint_t*': 'PyInt_FromLong',
    'uint_t': 'PyInt_FromLong',
}

def gen_new_init(newfunc, name):
    newparams = get_params_types_names(newfunc)
    # self->param1, self->param2, self->param3
    if len(newparams):
        selfparams = ', self->'+', self->'.join([p['name'] for p in newparams])
    else:
        selfparams = '' 
    # "param1", "param2", "param3"
    paramnames = ", ".join(["\""+p['name']+"\"" for p in newparams])
    pyparams = "".join(map(lambda p: aubio2pytypes[p['type']], newparams))
    paramrefs = ", ".join(["&" + p['name'] for p in newparams])
    s = """\
// WARNING: this file is generated, DO NOT EDIT

// WARNING: if you haven't read the first line yet, please do so
#include "aubiowraphell.h"

typedef struct
{
  PyObject_HEAD
  aubio_%(name)s_t * o;
""" % locals()
    for p in newparams:
        ptype = p['type']
        pname = p['name']
        s += """\
  %(ptype)s %(pname)s;
""" % locals()
    s += """\
} Py_%(name)s;

static char Py_%(name)s_doc[] = "%(name)s object";

static PyObject *
Py_%(name)s_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
  Py_%(name)s *self;
""" % locals()
    for p in newparams:
        ptype = p['type']
        pname = p['name']
        initval = aubioinitvalue[ptype]
        s += """\
  %(ptype)s %(pname)s = %(initval)s;
""" % locals()
    # now the actual PyArg_Parse
    if len(paramnames):
        s += """\
  static char *kwlist[] = { %(paramnames)s, NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|%(pyparams)s", kwlist,
          %(paramrefs)s)) {
    return NULL;
  }
""" % locals()
    s += """\

  self = (Py_%(name)s *) pytype->tp_alloc (pytype, 0);

  if (self == NULL) {
    return NULL;
  }
""" % locals()
    for p in newparams:
        ptype = p['type']
        pname = p['name']
        defval = aubiodefvalue[pname]
        if ptype == 'char_t*':
            s += """\

  self->%(pname)s = %(defval)s;
  if (%(pname)s != NULL) {
    self->%(pname)s = %(pname)s;
  }
""" % locals()
        elif ptype == 'uint_t':
            s += """\

  self->%(pname)s = %(defval)s;
  if ((sint_t)%(pname)s > 0) {
    self->%(pname)s = %(pname)s;
  } else if ((sint_t)%(pname)s < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative value for %(pname)s");
    return NULL;
  }
""" % locals()
        elif ptype == 'smpl_t':
            s += """\

  self->%(pname)s = %(defval)s;
  if (%(pname)s != %(defval)s) {
    self->%(pname)s = %(pname)s;
  }
""" % locals()
        else:
            write_msg ("ERROR, unknown type of parameter %s %s" % (ptype, pname) )
    s += """\

  return (PyObject *) self;
}

AUBIO_INIT(%(name)s %(selfparams)s)

AUBIO_DEL(%(name)s)

""" % locals()
    return s

def gen_do_input_params(inputparams):
  inputdefs = ''
  parseinput = ''
  inputrefs = ''
  inputvecs = ''
  pytypes = ''

  if len(inputparams):
    # build the parsing string for PyArg_ParseTuple
    pytypes = "".join([aubio2pytypes[p['type']] for p in inputparams])

    inputdefs = "  /* input vectors python prototypes */\n"
    for p in inputparams:
      if p['type'] != 'uint_t':
        inputdefs += "  PyObject * " + p['name'] + "_obj;\n"

    inputvecs = "  /* input vectors prototypes */\n  "
    inputvecs += "\n  ".join(map(lambda p: p['type'] + ' ' + p['name'] + ";", inputparams))

    parseinput = "  /* input vectors parsing */\n  "
    for p in inputparams:
        inputvec = p['name']
        if p['type'] != 'uint_t':
          inputdef = p['name'] + "_obj"
        else:
          inputdef = p['name']
        converter = aubiovecfrompyobj[p['type']]
        if p['type'] != 'uint_t':
          parseinput += """%(inputvec)s = %(converter)s (%(inputdef)s);

  if (%(inputvec)s == NULL) {
    return NULL;
  }

  """ % locals()

    # build the string for the input objects references
    inputreflist = []
    for p in inputparams:
      if p['type'] != 'uint_t':
        inputreflist += [ "&" + p['name'] + "_obj" ]
      else:
        inputreflist += [ "&" + p['name'] ]
    inputrefs = ", ".join(inputreflist)
    # end of inputs strings
  return inputdefs, parseinput, inputrefs, inputvecs, pytypes

def gen_do_output_params(outputparams, name):
  outputvecs = ""
  outputcreate = ""
  if len(outputparams):
    outputvecs = "  /* output vectors prototypes */\n"
    for p in outputparams:
      params = {
        'name': p['name'], 'pytype': p['type'], 'autype': p['type'][:-3],
        'length': defaultsizes[name].pop(0) }
      if (p['type'] == 'uint_t*'):
        outputvecs += '  uint_t' + ' ' + p['name'] + ";\n"
        outputcreate += "  %(name)s = 0;\n" % params
      else:
        outputvecs += "  " + p['type'] + ' ' + p['name'] + ";\n"
        outputcreate += "  /* creating output %(name)s as a new_%(autype)s of length %(length)s */\n" % params
        outputcreate += "  %(name)s = new_%(autype)s (%(length)s);\n" % params

  returnval = "";
  if len(outputparams) > 1:
    returnval += "  PyObject *outputs = PyList_New(0);\n"
    for p in outputparams:
      returnval += "  PyList_Append( outputs, (PyObject *)" + aubiovectopyobj[p['type']] + " (" + p['name'] + ")" +");\n"
    returnval += "  return outputs;"
  elif len(outputparams) == 1:
    if defaultsizes[name] == '1':
      returnval += "  return (PyObject *)PyFloat_FromDouble(" + p['name'] + "->data[0])"
    else:
      returnval += "  return (PyObject *)" + aubiovectopyobj[p['type']] + " (" + p['name'] + ")"
  else:
    returnval += "  Py_RETURN_NONE"
  # end of output strings
  return outputvecs, outputcreate, returnval

def gen_do(dofunc, name):
    funcname = dofunc.split()[1].split('(')[0]
    doparams = get_params_types_names(dofunc) 
    # make sure the first parameter is the object
    assert doparams[0]['type'] == "aubio_"+name+"_t*", \
        "method is not in 'aubio_<name>_t"
    # and remove it
    doparams = doparams[1:]

    n_param = len(doparams)

    if name in param_numbers.keys():
      n_input_param, n_output_param = param_numbers[name]
    else:
      n_input_param, n_output_param = 1, n_param - 1

    assert n_output_param + n_input_param == n_param, "n_output_param + n_input_param != n_param for %s" % name

    inputparams = doparams[:n_input_param]
    outputparams = doparams[n_input_param:n_input_param + n_output_param]

    inputdefs, parseinput, inputrefs, inputvecs, pytypes = gen_do_input_params(inputparams);
    outputvecs, outputcreate, returnval = gen_do_output_params(outputparams, name)

    # build strings for outputs
    # build the parameters for the  _do() call
    doparams_string = "self->o"
    for p in doparams:
      if p['type'] == 'uint_t*':
        doparams_string += ", &" + p['name']
      else:
        doparams_string += ", " + p['name']

    if n_input_param:
      arg_parse_tuple = """\
  if (!PyArg_ParseTuple (args, "%(pytypes)s", %(inputrefs)s)) {
    return NULL;
  }
""" % locals()
    else:
      arg_parse_tuple = ""
    # put it all together
    s = """\
/* function Py_%(name)s_do */
static PyObject * 
Py_%(name)s_do(Py_%(name)s * self, PyObject * args)
{
%(inputdefs)s
%(inputvecs)s
%(outputvecs)s

%(arg_parse_tuple)s

%(parseinput)s
  
%(outputcreate)s

  /* compute _do function */
  %(funcname)s (%(doparams_string)s);

%(returnval)s;
}
""" % locals()
    return s

def gen_members(new_method, name):
    newparams = get_params_types_names(new_method)
    s = """
AUBIO_MEMBERS_START(%(name)s)""" % locals()
    for param in newparams:
        if param['type'] == 'char_t*':
            s += """
  {"%(pname)s", T_STRING, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param['name'], 'ptype': param['type'], 'name': name}
        elif param['type'] == 'uint_t':
            s += """
  {"%(pname)s", T_INT, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param['name'], 'ptype': param['type'], 'name': name}
        elif param['type'] == 'smpl_t':
            s += """
  {"%(pname)s", T_FLOAT, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param['name'], 'ptype': param['type'], 'name': name}
        else:
            write_msg ("-- ERROR, unknown member type ", param )
    s += """
AUBIO_MEMBERS_STOP(%(name)s)

""" % locals()
    return s


def gen_methods(get_methods, set_methods, name):
    s = ""
    method_defs = ""
    for method in set_methods:
        method_name = get_name(method)
        params = get_params_types_names(method)
        out_type = get_return_type(method)
        assert params[0]['type'] == "aubio_"+name+"_t*", \
            "get method is not in 'aubio_<name>_t"
        write_msg (method )
        write_msg (params[1:])
        setter_args = "self->o, " +",".join([p['name'] for p in params[1:]])
        parse_args = ""
        for p in params[1:]:
            parse_args += p['type'] + " " + p['name'] + ";\n"
        argmap = "".join([aubio2pytypes[p['type']] for p in params[1:]])
        arglist = ", ".join(["&"+p['name'] for p in params[1:]])
        parse_args += """
  if (!PyArg_ParseTuple (args, "%(argmap)s", %(arglist)s)) {
    return NULL;
  } """ % locals()
        s += """
static PyObject *
Py%(funcname)s (Py_%(objname)s *self, PyObject *args)
{
  uint_t err = 0;

  %(parse_args)s

  err = %(funcname)s (%(setter_args)s);

  if (err > 0) {
    PyErr_SetString (PyExc_ValueError,
        "error running %(funcname)s");
    return NULL;
  }
  Py_RETURN_NONE;
}
""" % {'funcname': method_name, 'objname': name, 
        'out_type': out_type, 'setter_args': setter_args, 'parse_args': parse_args }
        shortname = method_name.split('aubio_'+name+'_')[-1]
        method_defs += """\
  {"%(shortname)s", (PyCFunction) Py%(method_name)s,
    METH_VARARGS, ""},
""" % locals()

    for method in get_methods:
        method_name = get_name(method)
        params = get_params_types_names(method)
        out_type = get_return_type(method)
        assert params[0]['type'] == "aubio_"+name+"_t*", \
            "get method is not in 'aubio_<name>_t %s" % params[0]['type']
        assert len(params) == 1, \
            "get method has more than one parameter %s" % params
        getter_args = "self->o" 
        returnval = "(PyObject *)" + aubiovectopyobj[out_type] + " (tmp)"
        shortname = method_name.split('aubio_'+name+'_')[-1]
        method_defs += """\
  {"%(shortname)s", (PyCFunction) Py%(method_name)s,
    METH_NOARGS, ""},
""" % locals()
        s += """
static PyObject *
Py%(funcname)s (Py_%(objname)s *self, PyObject *unused)
{
  %(out_type)s tmp = %(funcname)s (%(getter_args)s);
  return %(returnval)s;
}
""" % {'funcname': method_name, 'objname': name, 
        'out_type': out_type, 'getter_args': getter_args, 'returnval': returnval }

    s += """
static PyMethodDef Py_%(name)s_methods[] = {
""" % locals() 
    s += method_defs 
    s += """\
  {NULL} /* sentinel */
};
""" % locals() 
    return s

def gen_finish(name):
    s = """\

AUBIO_TYPEOBJECT(%(name)s, "aubio.%(name)s")
""" % locals()
    return s

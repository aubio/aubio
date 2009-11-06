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

# TODO
# do function: for now, only the following pattern is supported:
# void aubio_<foo>_do (aubio_foo_t * o, 
#       [input1_t * input, [output1_t * output, ..., output3_t * output]]);
# There is no way of knowing that output1 is actually input2. In the future,
# const could be used for the inputs in the C prototypes.

# the important bits: the size of the output for each objects. this data should
# move into the C library at some point.
defaultsizes = {
    'resampler':    ('input->length * self->ratio', 'input->channels'),
    'specdesc':     ('1', 'self->channels'),
    'onset':        ('1', 'self->channels'),
    'pitchyin':     ('1', 'in->channels'),
    'pitchyinfft':  ('1', 'in->channels'),
    'pitchschmitt': ('1', 'in->channels'),
    'pitchmcomb':   ('1', 'self->channels'),
    'pitchfcomb':   ('1', 'self->channels'),
    'pitch':        ('1', 'self->channels'),
    'tss':          ('self->hop_size', 'self->channels'),
    'mfcc':         ('self->n_coeffs', 'in->channels'),
    'beattracking': ('self->hop_size', 'self->channels'),
    'tempo':        ('1', 'self->channels'),
    'peakpicker':   ('1', 'self->channels'),
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
    'channels': 'Py_default_vector_channels', 
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
    }

# aubio to python
aubio2pytypes = {
    'uint_t': 'I',
    'smpl_t': 'I',
    'lsmp_t': 'I',
    'fvec_t': 'O',
    'cvec_t': 'O',
    'char_t*': 's',
}

# aubio to pyaubio
aubio2pyaubio = {
    'fvec_t': 'Py_fvec',
    'cvec_t': 'Py_cvec',
}

# array to aubio
aubiovecfrompyobj = {
    'fvec_t': 'PyAubio_ArrayToFvec',
    'cvec_t': 'PyAubio_ArrayToCvec',
}

# aubio to array
aubiovectopyobj = {
    'fvec_t': 'PyAubio_FvecToArray',
    'cvec_t': 'PyAubio_CvecToArray',
}

def get_newparams(newfunc):
    newparams = [[p.split()[0], p.split()[-1]]
            for p in newfunc.split('(')[1].split(')')[0].split(',')]
    # make char_t a pointer 
    return map(lambda x: [x[0].replace('char_t', 'char_t*'), x[1]], newparams)

def gen_new_init(newfunc, name):
    newparams = get_newparams(newfunc)
    # self->param1, self->param2, self->param3
    selfparams = ', self->'.join([p[1] for p in newparams])
    # "param1", "param2", "param3"
    paramnames = ", ".join(["\""+p[1]+"\"" for p in newparams])
    pyparams = "".join(map(lambda p: aubio2pytypes[p[0]], newparams))
    paramrefs = ", ".join(["&" + p[1] for p in newparams])
    s = """\
// WARNING: this file is generated, DO NOT EDIT
#include "aubiowraphell.h"

typedef struct
{
  PyObject_HEAD
  aubio_%(name)s_t * o;
""" % locals()
    for ptype, pname in newparams:
        s += """\
  %(ptype)s %(pname)s;
""" % locals()
    s += """\
} Py_%(name)s;

static char Py_%(name)s_doc[] = "%(name)s object";

static PyObject *
Py_%(name)s_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{
""" % locals()
    for ptype, pname in newparams:
        initval = aubioinitvalue[ptype]
        s += """\
  %(ptype)s %(pname)s = %(initval)s;
""" % locals()
    # now the actual PyArg_Parse
    s += """\
  Py_%(name)s *self;
  static char *kwlist[] = { %(paramnames)s, NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|%(pyparams)s", kwlist,
          %(paramrefs)s)) {
    return NULL;
  }

  self = (Py_%(name)s *) pytype->tp_alloc (pytype, 0);

  if (self == NULL) {
    return NULL;
  }
""" % locals()
    # TODO add parameters default values
    for ptype, pname in newparams:
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
  if (%(pname)s > 0) {
    self->%(pname)s = %(pname)s;
  } else if (%(pname)s < 0) {
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
            print "ERROR, unknown type of parameter %s %s" % (ptype, pname)
    s += """\

  return (PyObject *) self;
}

AUBIO_INIT(%(name)s, self->%(selfparams)s)

AUBIO_DEL(%(name)s)

""" % locals()
    return s

def gen_do(dofunc, name):
    funcname = dofunc.split()[1].split('(')[0]
    doparams = [p.split() for p in dofunc.split('(')[1].split(')')[0].split(',')]
    # make sure the first parameter is the object
    assert doparams[0][0] == "aubio_"+name+"_t", \
        "method is not in 'aubio_<name>_t"
    # and remove it
    doparams = doparams[1:]
    # guess the input/output params, assuming we have less than 3
    assert len(doparams) > 0, \
        "no parameters for function do in object %s" % name
    #assert (len(doparams) <= 2), \
    #    "more than 3 parameters for do in object %s" % name

    # build strings for inputs, assuming there is only one input 
    inputparams = [doparams[0]]
    # build the parsing string for PyArg_ParseTuple
    pytypes = "".join([aubio2pytypes[p[0]] for p in doparams[0:1]])
    inputdefs = "\n  ".join(["PyObject * " + p[-1] + "_obj;" for p in inputparams])
    inputvecs = "\n  ".join(map(lambda p: \
                aubio2pyaubio[p[0]]+" * " + p[-1] + ";", inputparams))
    parseinput = ""
    for p in inputparams:
        inputvec = p[-1]
        inputdef = p[-1] + "_obj"
        converter = aubiovecfrompyobj[p[0]]
        parseinput += """%(inputvec)s = %(converter)s (%(inputdef)s);

  if (%(inputvec)s == NULL) {
    return NULL;
  }""" % locals()
    # build the string for the input objects references
    inputrefs = ", ".join(["&" + p[-1] + "_obj" for p in inputparams])
    # end of inputs strings

    # build strings for outputs
    outputparams = doparams[1:]
    if len(doparams) > 1:
        #assert len(outputparams) == 1, \
        #    "too many output parameters"
        outputvecs = "\n  ".join([aubio2pyaubio[p[0]]+" * " + p[-1] + ";" for p in outputparams])
        outputcreate = """\
AUBIO_NEW_VEC(%(name)s, %(pytype)s, %(length)s, %(channels)s)
  %(name)s->o = new_%(autype)s (%(length)s, %(channels)s);""" % \
    {'name': p[-1], 'pytype': aubio2pyaubio[p[0]], 'autype': p[0][:-2],
        'length': defaultsizes[name][0], 'channels': defaultsizes[name][1]}
        returnval = "(PyObject *)" + aubiovectopyobj[p[0]] + " (" + p[-1] + ")"
    else:
        # no output
        outputvecs = ""
        outputcreate = ""
        #returnval = "Py_None";
        returnval = "(PyObject *)" + aubiovectopyobj[p[0]] + " (" + p[-1] + ")"
    # end of output strings

    # build the parameters for the  _do() call
    doparams_string = "self->o, " + ", ".join([p[-1]+"->o" for p in doparams])

    # put it all together
    s = """\
static PyObject * 
Py_%(name)s_do(Py_%(name)s * self, PyObject * args)
{
  %(inputdefs)s
  %(inputvecs)s
  %(outputvecs)s

  if (!PyArg_ParseTuple (args, "%(pytypes)s", %(inputrefs)s)) {
    return NULL;
  }

  %(parseinput)s
  
  %(outputcreate)s

  /* compute _do function */
  %(funcname)s (%(doparams_string)s);

  return %(returnval)s;
}
""" % locals()
    return s

def gen_members(new_method, name):
    newparams = get_newparams(new_method)
    s = """
AUBIO_MEMBERS_START(%(name)s)""" % locals()
    for param in newparams:
        if param[0] == 'char_t*':
            s += """
  {"%(pname)s", T_STRING, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param[1], 'ptype': param[0], 'name': name}
        elif param[0] == 'uint_t':
            s += """
  {"%(pname)s", T_INT, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param[1], 'ptype': param[0], 'name': name}
        elif param[0] == 'smpl_t':
            s += """
  {"%(pname)s", T_FLOAT, offsetof (Py_%(name)s, %(pname)s), READONLY, ""},""" \
        % { 'pname': param[1], 'ptype': param[0], 'name': name}
        else:
            print "-- ERROR, unknown member type ", param
    s += """
AUBIO_MEMBERS_STOP(%(name)s)

""" % locals()
    return s

def gen_methods(get_methods, set_methods, name):
    # TODO add methods 
    s = """\
static PyMethodDef Py_%(name)s_methods[] = {
""" % locals() 
    # TODO add PyMethodDefs
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

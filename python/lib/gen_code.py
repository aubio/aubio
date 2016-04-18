aubiodefvalue = {
    # we have some clean up to do
    'buf_size': 'Py_default_vector_length',
    'win_s': 'Py_default_vector_length',
    # and here too
    'hop_size': 'Py_default_vector_length / 2',
    'hop_s': 'Py_default_vector_length / 2',
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

member_types = {
        'name': 'type',
        'char_t*': 'T_STRING',
        'uint_t': 'T_INT',
        'smpl_t': 'T_FLOAT',
        }

pyfromtype_fn = {
        'smpl_t': 'PyFloat_FromDouble',
        'uint_t': 'PyLong_FromLong', # was: 'PyInt_FromLong',
        'fvec_t*': 'PyAubio_CFvecToArray',
        'fmat_t*': 'PyAubio_CFmatToArray',
        }

pytoaubio_fn = {
        'fvec_t*': 'PyAubio_ArrayToCFvec',
        'cvec_t*': 'PyAubio_ArrayToCCvec',
        'fmat_t*': 'PyAubio_ArrayToCFmat',
        }

pyfromaubio_fn = {
        'fvec_t*': 'PyAubio_CFvecToArray',
        'cvec_t*': 'PyAubio_CCvecToArray',
        'fmat_t*': 'PyAubio_CFmatToArray',
        }

newfromtype_fn = {
        'fvec_t*': 'new_fvec',
        'fmat_t*': 'new_fmat',
        'cvec_t*': 'new_cvec',
        }

delfromtype_fn = {
        'fvec_t*': 'del_fvec',
        'fmat_t*': 'del_fmat',
        'cvec_t*': 'del_cvec',
        }

param_init = {
        'char_t*': 'NULL',
        'uint_t': '0',
        'sint_t': 0,
        'smpl_t': 0.,
        'lsmp_t': 0.,
        }

pyargparse_chars = {
        'smpl_t': 'f',
        'uint_t': 'I',
        'sint_t': 'I',
        'char_t*': 's',
        'fmat_t*': 'O',
        'fvec_t*': 'O',
        }

objoutsize = {
        'onset': '1',
        'pitch': '1',
        'wavetable': 'self->hop_size',
        'sampler': 'self->hop_size',
        'mfcc': 'self->n_coeffs',
        'specdesc': '1',
        'tempo': '1',
        'filterbank': 'self->n_filters',
        }

def get_name(proto):
    name = proto.replace(' *', '* ').split()[1].split('(')[0]
    name = name.replace('*','')
    if name == '': raise ValueError(proto + "gave empty name")
    return name

def get_return_type(proto):
    import re
    paramregex = re.compile('(\w+ ?\*?).*')
    outputs = paramregex.findall(proto)
    assert len(outputs) == 1
    return outputs[0].replace(' ', '')

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
    return list(map(split_type, get_params(proto)))


class MappedObject(object):

    def __init__(self, prototypes):
        self.prototypes = prototypes

        self.shortname = prototypes['shortname']
        self.longname = prototypes['longname']
        self.new_proto = prototypes['new'][0]
        self.del_proto = prototypes['del'][0]
        self.do_proto = prototypes['do'][0]
        self.input_params = get_params_types_names(self.new_proto)
        self.input_params_list = "; ".join(get_params(self.new_proto))
        self.outputs = get_params_types_names(self.do_proto)[2:]
        self.outputs_flat = get_params(self.do_proto)[2:]
        self.output_results = ", ".join(self.outputs_flat)

    def gen_code(self):
        out = ""
        out += self.gen_struct()
        out += self.gen_doc()
        out += self.gen_new()
        out += self.gen_init()
        out += self.gen_del()
        out += self.gen_do()
        out += self.gen_memberdef()
        out += self.gen_set()
        out += self.gen_get()
        out += self.gen_methodef()
        out += self.gen_typeobject()
        return out

    def gen_struct(self):
        out = """
// {shortname} structure
typedef struct{{
    PyObject_HEAD
    // pointer to aubio object
    {longname} *o;
    // input parameters
    {input_params_list};
    // output results
    {output_results};
}} Py_{shortname};
"""
        return out.format(**self.__dict__)

    def gen_doc(self):
        out = """
// TODO: add documentation
static char Py_{shortname}_doc[] = \"undefined\";
    """
        return out.format(**self.__dict__)

    def gen_new(self):
        out = """
// new {shortname}
static PyObject *
Py_{shortname}_new (PyTypeObject * pytype, PyObject * args, PyObject * kwds)
{{
    Py_{shortname} *self;
""".format(**self.__dict__)
        params = self.input_params
        for p in params:
            out += """
    {type} {name} = {defval};""".format(defval = param_init[p['type']], **p)
        plist = ", ".join(["\"%s\"" % p['name'] for p in params])
        out += """
    static char *kwlist[] = {{ {plist}, NULL }};""".format(plist = plist)
        argchars = "".join([pyargparse_chars[p['type']] for p in params])
        arglist = ", ".join(["&%s" % p['name'] for p in params])
        out += """
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|{argchars}", kwlist,
              {arglist})) {{
        return NULL;
    }}
""".format(argchars = argchars, arglist = arglist)
        out += """
    self = (Py_{shortname} *) pytype->tp_alloc (pytype, 0);
    if (self == NULL) {{
        return NULL;
    }}
""".format(**self.__dict__)
        params = self.input_params
        for p in params:
            out += self.check_valid(p) 
        out += """
    return (PyObject *)self;
}
"""
        return out

    def check_valid(self, p):
        if p['type'] == 'uint_t':
            return self.check_valid_uint(p)
        if p['type'] == 'char_t*':
            return self.check_valid_char(p)
        else:
            print ("ERROR, no idea how to check %s for validity" % p['type'])

    def check_valid_uint(self, p):
        name = p['name']
        return """
    self->{name} = {defval};
    if ((sint_t){name} > 0) {{
        self->{name} = {name};
    }} else if ((sint_t){name} < 0) {{
        PyErr_SetString (PyExc_ValueError, "can not use negative value for {name}");
        return NULL;
    }}
""".format(defval = aubiodefvalue[name], name = name)

    def check_valid_char(self, p):
        name = p['name']
        return """
    self->{name} = {defval};
    if ({name} != NULL) {{
        self->{name} = {name};
    }}
""".format(defval = aubiodefvalue[name], name = name)

    def gen_init(self):
        out = """
// init {shortname}
static int
Py_{shortname}_init (Py_{shortname} * self, PyObject * args, PyObject * kwds)
{{
""".format(**self.__dict__)
        new_name = get_name(self.new_proto)
        new_params = ", ".join(["self->%s" % s['name'] for s in self.input_params])
        out += """
  self->o = {new_name}({new_params});
""".format(new_name = new_name, new_params = new_params)
        paramchars = "%s"
        paramvals = "self->method"
        out += """
  // return -1 and set error string on failure
  if (self->o == NULL) {{
    //char_t errstr[30 + strlen(self->uri)];
    //sprintf(errstr, "error creating {shortname} with params {paramchars}", {paramvals});
    char_t errstr[60];
    sprintf(errstr, "error creating {shortname} with given params");
    PyErr_SetString (PyExc_Exception, errstr);
    return -1;
  }}
""".format(paramchars = paramchars, paramvals = paramvals, **self.__dict__)
        output_create = ""
        for o in self.outputs:
            output_create += """
  self->{name} = {create_fn}({output_size});""".format(name = o['name'], create_fn = newfromtype_fn[o['type']], output_size = objoutsize[self.shortname])
        out += """
  // TODO get internal params after actual object creation?
"""
        out += """
  // create outputs{output_create}
""".format(output_create = output_create)
        out += """
  return 0;
}
"""
        return out

    def gen_memberdef(self):
        out = """
static PyMemberDef Py_{shortname}_members[] = {{
""".format(**self.__dict__)
        for p in get_params_types_names(self.new_proto):
            tmp = "  {{\"{name}\", {ttype}, offsetof (Py_{shortname}, {name}), READONLY, \"TODO documentation\"}},\n"
            pytype = member_types[p['type']]
            out += tmp.format(name = p['name'], ttype = pytype, shortname = self.shortname)
        out += """  {NULL}, // sentinel
};
"""
        return out

    def gen_del(self):
        out = """
// del {shortname}
static void
Py_{shortname}_del  (Py_{shortname} * self, PyObject * unused)
{{""".format(**self.__dict__)
        for o in self.outputs:
            name = o['name']
            del_out = delfromtype_fn[o['type']]
            out += """
    {del_out}(self->{name});""".format(del_out = del_out, name = name)
        del_fn = get_name(self.del_proto)
        out += """
    {del_fn}(self->o);
    Py_TYPE(self)->tp_free((PyObject *) self);
}}
""".format(del_fn = del_fn)
        return out

    def gen_do(self):
        do_fn = get_name(self.do_proto)
        input_param = get_params_types_names(self.do_proto)[1];
        pytoaubio = pytoaubio_fn[input_param['type']]
        output = self.outputs[0]
        out = """
// do {shortname}
static PyObject*
Py_{shortname}_do  (Py_{shortname} * self, PyObject * args)
{{
    PyObject * in_obj;
    {input_type} {input_name};

    if (!PyArg_ParseTuple (args, "O", &in_obj)) {{
        return NULL;
    }}
    {input_name} = {pytoaubio} (in_obj); 
    if ({input_name} == NULL) {{
        return NULL;
    }}

    {do_fn}(self->o, {input_name}, {outputs});

    return (PyObject *) {aubiotonumpy} ({outputs});
}}
"""
        return out.format(do_fn = do_fn,
                shortname = self.prototypes['shortname'],
                input_name = input_param['name'],
                input_type= input_param['type'],
                pytoaubio = pytoaubio,
                outputs = ", ".join(["self->%s" % p['name'] for p in self.outputs]),
                aubiotonumpy = pyfromaubio_fn[output['type']], 
                )

    def gen_set(self):
        out = """
// {shortname} setters
""".format(**self.__dict__)
        for set_param in self.prototypes['set']:
            params = get_params_types_names(set_param)[1]
            paramtype = params['type']
            method_name = get_name(set_param)
            param = method_name.split('aubio_'+self.shortname+'_set_')[-1]
            pyparamtype = pyargparse_chars[paramtype]
            out += """
static PyObject *
Pyaubio_{shortname}_set_{param} (Py_{shortname} *self, PyObject *args)
{{
  uint_t err = 0;
  {paramtype} {param};

  if (!PyArg_ParseTuple (args, "{pyparamtype}", &{param})) {{
    return NULL;
  }}
  err = aubio_{shortname}_set_{param} (self->o, {param});

  if (err > 0) {{
    PyErr_SetString (PyExc_ValueError, "error running aubio_{shortname}_set_{param}");
    return NULL;
  }}
  Py_RETURN_NONE;
}}
""".format(param = param, paramtype = paramtype, pyparamtype = pyparamtype, **self.__dict__)
        return out

    def gen_get(self):
        out = """
// {shortname} getters
""".format(**self.__dict__)
        for method in self.prototypes['get']:
            params = get_params_types_names(method)
            method_name = get_name(method)
            assert len(params) == 1, \
                "get method has more than one parameter %s" % params
            param = method_name.split('aubio_'+self.shortname+'_get_')[-1]
            paramtype = get_return_type(method)
            ptypeconv = pyfromtype_fn[paramtype]
            out += """
static PyObject *
Pyaubio_{shortname}_get_{param} (Py_{shortname} *self, PyObject *unused)
{{
  {ptype} {param} = aubio_{shortname}_get_{param} (self->o);
  return (PyObject *){ptypeconv} ({param});
}}
""".format(param = param, ptype = paramtype, ptypeconv = ptypeconv,
        **self.__dict__)
        return out

    def gen_methodef(self):
        out = """
static PyMethodDef Py_{shortname}_methods[] = {{""".format(**self.__dict__)
        for m in self.prototypes['set']:
            name = get_name(m)
            shortname = name.replace('aubio_%s_' % self.shortname, '')
            out += """
  {{"{shortname}", (PyCFunction) Py{name},
    METH_VARARGS, ""}},""".format(name = name, shortname = shortname)
        for m in self.prototypes['get']:
            name = get_name(m)
            shortname = name.replace('aubio_%s_' % self.shortname, '')
            out += """
  {{"{shortname}", (PyCFunction) Py{name},
    METH_NOARGS, ""}},""".format(name = name, shortname = shortname)
        out += """
  {NULL} /* sentinel */
};
"""
        return out

    def gen_typeobject(self):
        return """
PyTypeObject Py_{shortname}Type = {{
  //PyObject_HEAD_INIT (NULL)
  //0,
  PyVarObject_HEAD_INIT (NULL, 0)
  "aubio.{shortname}",
  sizeof (Py_{shortname}),
  0,
  (destructor) Py_{shortname}_del,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  (ternaryfunc)Py_{shortname}_do,
  0,
  0,
  0,
  0,
  Py_TPFLAGS_DEFAULT,
  Py_{shortname}_doc,
  0,
  0,
  0,
  0,
  0,
  0,
  Py_{shortname}_methods,
  Py_{shortname}_members,
  0,
  0,
  0,
  0,
  0,
  0,
  (initproc) Py_{shortname}_init,
  0,
  Py_{shortname}_new,
}};
""".format(**self.__dict__)

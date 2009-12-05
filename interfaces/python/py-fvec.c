#include "aubio-types.h"

/* fvec type definition 

class fvec():
    def __init__(self, length = 1024):
        self.length = length 
        self.data = array(length)

*/

static char Py_fvec_doc[] = "fvec object";

static PyObject *
Py_fvec_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int length= 0;
  Py_fvec *self;
  static char *kwlist[] = { "length", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|I", kwlist,
          &length)) {
    return NULL;
  }

  self = (Py_fvec *) type->tp_alloc (type, 0);

  self->length = Py_default_vector_length;

  if (self == NULL) {
    return NULL;
  }

  if (length > 0) {
    self->length = length;
  } else if (length < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of elements");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_fvec_init (Py_fvec * self, PyObject * args, PyObject * kwds)
{
  self->o = new_fvec (self->length);
  if (self->o == NULL) {
    return -1;
  }

  return 0;
}

static void
Py_fvec_del (Py_fvec * self)
{
  del_fvec (self->o);
  self->ob_type->tp_free ((PyObject *) self);
}

static PyObject *
Py_fvec_repr (Py_fvec * self, PyObject * unused)
{
#if 0
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyString_FromString ("aubio fvec of %d elements");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("I", self->length);
  if (args == NULL) {
    goto fail;
  }
  fvec_print ( self->o );

  result = PyString_Format (format, args);

fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
#endif
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyString_FromString ("%s");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("O", PyAubio_FvecToArray (self));
  if (args == NULL) {
    goto fail;
  }

  result = PyString_Format (format, args);
fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
}

Py_fvec *
PyAubio_ArrayToFvec (PyObject *input) {
  PyObject *array;
  Py_fvec *vec;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  // parsing input object into a Py_fvec
  if (PyObject_TypeCheck (input, &Py_fvecType)) {
    // input is an fvec, nothing else to do
    vec = (Py_fvec *) input;
  } else if (PyArray_Check(input)) {

    // we got an array, convert it to an fvec 
    if (PyArray_NDIM (input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM (input) > 1) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than one dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT (input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
    } else if (PyArray_TYPE (input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    } else {
      // input data type is float32, nothing else to do
      array = input;
    }

    // create a new fvec object
    vec = (Py_fvec*) PyObject_New (Py_fvec, &Py_fvecType); 
    vec->length = PyArray_SIZE (array);

    // no need to really allocate fvec, just its struct member 
    // vec->o = new_fvec (vec->length);
    vec->o = (fvec_t *)malloc(sizeof(fvec_t));
    vec->o->length = vec->length;
    vec->o->data = (smpl_t *) PyArray_GETPTR1 (array, 0);

  } else if (PyObject_TypeCheck (input, &PyList_Type)) {
    PyErr_SetString (PyExc_ValueError, "does not convert from list yet");
    return NULL;
  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept vector or fvec as input");
    return NULL;
  }

  return vec;

fail:
  return NULL;
}

PyObject *
PyAubio_CFvecToArray (fvec_t * self)
{
  npy_intp dims[] = { self->length, 1 };
  return  PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->data);
}

PyObject *
PyAubio_FvecToArray (Py_fvec * self)
{
  PyObject *array = NULL;
  npy_intp dims[] = { self->length, 1 };
  array = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, self->o->data);
  return array;
}

static PyObject *
Py_fvec_getitem (Py_fvec * self, Py_ssize_t index)
{
  if (index < 0 || index >= self->length) {
    PyErr_SetString (PyExc_IndexError, "no such element");
    return NULL;
  }

  return PyFloat_FromDouble (self->o->data[index]);
}

static int
Py_fvec_setitem (Py_fvec * self, Py_ssize_t index, PyObject * o)
{

  if (index < 0 || index >= self->length) {
    PyErr_SetString (PyExc_IndexError, "no such element");
    goto fail;
  }

  if (PyFloat_Check (o)) {
    PyErr_SetString (PyExc_ValueError, "should be a float");
    goto fail;
  }

  self->o->data[index] = (smpl_t) PyFloat_AsDouble(o);

  return 0;

fail:
  return -1;
}

int
Py_fvec_get_length (Py_fvec * self)                                                                                                                                     
{                                                                                                                                                                        
  return self->length;                                                                                                                                                 
}

static PyMemberDef Py_fvec_members[] = {
  // TODO remove READONLY flag and define getter/setter
  {"length", T_INT, offsetof (Py_fvec, length), READONLY,
      "length attribute"},
  {NULL}                        /* Sentinel */
};

static PyMethodDef Py_fvec_methods[] = {
  {"__array__", (PyCFunction) PyAubio_FvecToArray, METH_NOARGS,
      "Returns the vector as a numpy array."},
  {NULL}
};

static PySequenceMethods Py_fvec_tp_as_sequence = {
  (lenfunc) Py_fvec_get_length,        /* sq_length         */
  0,                                    /* sq_concat         */
  0,                                    /* sq_repeat         */
  (ssizeargfunc) Py_fvec_getitem,       /* sq_item           */
  0,                                    /* sq_slice          */
  (ssizeobjargproc) Py_fvec_setitem,    /* sq_ass_item       */
  0,                                    /* sq_ass_slice      */
  0,                                    /* sq_contains       */
  0,                                    /* sq_inplace_concat */
  0,                                    /* sq_inplace_repeat */
};


PyTypeObject Py_fvecType = {
  PyObject_HEAD_INIT (NULL)
  0,                            /* ob_size           */
  "aubio.fvec",                 /* tp_name           */
  sizeof (Py_fvec),             /* tp_basicsize      */
  0,                            /* tp_itemsize       */
  (destructor) Py_fvec_del,     /* tp_dealloc        */
  0,                            /* tp_print          */
  0,                            /* tp_getattr        */
  0,                            /* tp_setattr        */
  0,                            /* tp_compare        */
  (reprfunc) Py_fvec_repr,      /* tp_repr           */
  0,                            /* tp_as_number      */
  &Py_fvec_tp_as_sequence,      /* tp_as_sequence    */
  0,                            /* tp_as_mapping     */
  0,                            /* tp_hash           */
  0,                            /* tp_call           */
  0,                            /* tp_str            */
  0,                            /* tp_getattro       */
  0,                            /* tp_setattro       */
  0,                            /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT,           /* tp_flags          */
  Py_fvec_doc,                  /* tp_doc            */
  0,                            /* tp_traverse       */
  0,                            /* tp_clear          */
  0,                            /* tp_richcompare    */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter           */
  0,                            /* tp_iternext       */
  Py_fvec_methods,              /* tp_methods        */
  Py_fvec_members,              /* tp_members        */
  0,                            /* tp_getset         */
  0,                            /* tp_base           */
  0,                            /* tp_dict           */
  0,                            /* tp_descr_get      */
  0,                            /* tp_descr_set      */
  0,                            /* tp_dictoffset     */
  (initproc) Py_fvec_init,      /* tp_init           */
  0,                            /* tp_alloc          */
  Py_fvec_new,                  /* tp_new            */
};

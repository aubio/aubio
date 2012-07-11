#include "aubio-types.h"

/* cvec type definition 

class cvec():
    def __init__(self, length = 1024):
        self.length = length 
        self.norm = array(length)
        self.phas = array(length)

*/

static char Py_cvec_doc[] = "cvec object";

static PyObject *
Py_cvec_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int length= 0;
  Py_cvec *self;
  static char *kwlist[] = { "length", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|I", kwlist,
          &length)) {
    return NULL;
  }


  self = (Py_cvec *) type->tp_alloc (type, 0);

  self->length = Py_default_vector_length / 2 + 1;

  if (self == NULL) {
    return NULL;
  }

  if (length > 0) {
    self->length = length / 2 + 1;
  } else if (length < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative number of elements");
    return NULL;
  }

  return (PyObject *) self;
}

static int
Py_cvec_init (Py_cvec * self, PyObject * args, PyObject * kwds)
{
  self->o = new_cvec ((self->length - 1) * 2);
  if (self->o == NULL) {
    return -1;
  }

  return 0;
}

static void
Py_cvec_del (Py_cvec * self)
{
  del_cvec (self->o);
  self->ob_type->tp_free ((PyObject *) self);
}

static PyObject *
Py_cvec_repr (Py_cvec * self, PyObject * unused)
{
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyString_FromString ("aubio cvec of %d elements");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("I", self->length);
  if (args == NULL) {
    goto fail;
  }
  cvec_print ( self->o );

  result = PyString_Format (format, args);

fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
}

PyObject *
PyAubio_CvecNormToArray (Py_cvec * self)
{
  npy_intp dims[] = { self->o->length, 1 };
  return PyArray_SimpleNewFromData (1, dims, NPY_FLOAT, self->o->norm);
}


PyObject *
PyAubio_CvecPhasToArray (Py_cvec * self)
{
  npy_intp dims[] = { self->o->length, 1 };
  return PyArray_SimpleNewFromData (1, dims, NPY_FLOAT, self->o->phas);
}

PyObject *
PyAubio_ArrayToCvecPhas (PyObject * self)
{
  return NULL;
}

PyObject *
Py_cvec_get_norm (Py_cvec * self, void *closure)
{
  return PyAubio_CvecNormToArray(self);
}

PyObject *
Py_cvec_get_phas (Py_cvec * self, void *closure)
{
  return PyAubio_CvecPhasToArray(self);
}

static int
Py_cvec_set_norm (Py_cvec * vec, PyObject *input, void * closure)
{
  PyArrayObject * array;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  if (PyArray_Check(input)) {

    // we got an array, convert it to a cvec.norm 
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM ((PyArrayObject *)input) > 2) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than two dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    }
    array = (PyArrayObject *)input;

    // check input array dimensions
    if (PyArray_NDIM (array) != 1) {
      PyErr_Format (PyExc_ValueError,
          "input array has %d dimensions, not 1",
          PyArray_NDIM (array));
      goto fail;
    } else {
      if (vec->o->length != PyArray_SIZE (array)) {
          PyErr_Format (PyExc_ValueError,
                  "input array has length %d, but cvec has length %d",
                  (int)PyArray_SIZE (array), vec->o->length);
          goto fail;
      }
    }

    vec->o->norm = (smpl_t *) PyArray_GETPTR1 (array, 0);

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array as input");
    return 1;
  }

  Py_INCREF(array);
  return 0;

fail:
  return 1;
}

static int
Py_cvec_set_phas (Py_cvec * vec, PyObject *input, void * closure)
{
  PyArrayObject * array;
  if (input == NULL) {
    PyErr_SetString (PyExc_ValueError, "input array is not a python object");
    goto fail;
  }
  if (PyArray_Check(input)) {

    // we got an array, convert it to a cvec.phas
    if (PyArray_NDIM ((PyArrayObject *)input) == 0) {
      PyErr_SetString (PyExc_ValueError, "input array is a scalar");
      goto fail;
    } else if (PyArray_NDIM ((PyArrayObject *)input) > 2) {
      PyErr_SetString (PyExc_ValueError,
          "input array has more than two dimensions");
      goto fail;
    }

    if (!PyArray_ISFLOAT ((PyArrayObject *)input)) {
      PyErr_SetString (PyExc_ValueError, "input array should be float");
      goto fail;
    } else if (PyArray_TYPE ((PyArrayObject *)input) != AUBIO_NPY_SMPL) {
      PyErr_SetString (PyExc_ValueError, "input array should be float32");
      goto fail;
    }
    array = (PyArrayObject *)input;

    // check input array dimensions
    if (PyArray_NDIM (array) != 1) {
      PyErr_Format (PyExc_ValueError,
          "input array has %d dimensions, not 1",
          PyArray_NDIM (array));
      goto fail;
    } else {
      if (vec->o->length != PyArray_SIZE (array)) {
          PyErr_Format (PyExc_ValueError,
                  "input array has length %d, but cvec has length %d",
                  (int)PyArray_SIZE (array), vec->o->length);
          goto fail;
      }
    }

    vec->o->phas = (smpl_t *) PyArray_GETPTR1 (array, 0);

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array as input");
    return 1;
  }

  Py_INCREF(array);
  return 0;

fail:
  return 1;
}

static PyMemberDef Py_cvec_members[] = {
  // TODO remove READONLY flag and define getter/setter
  {"length", T_INT, offsetof (Py_cvec, length), READONLY,
      "length attribute"},
  {NULL}                        /* Sentinel */
};

static PyMethodDef Py_cvec_methods[] = {
  {NULL}
};

static PyGetSetDef Py_cvec_getseters[] = {
  {"norm", (getter)Py_cvec_get_norm, (setter)Py_cvec_set_norm, 
      "Numpy vector of shape (length,) containing the magnitude",
      NULL},
  {"phas", (getter)Py_cvec_get_phas, (setter)Py_cvec_set_phas, 
      "Numpy vector of shape (length,) containing the phase",
      NULL},
  {NULL} /* sentinel */
};

PyTypeObject Py_cvecType = {
  PyObject_HEAD_INIT (NULL)
  0,                            /* ob_size           */
  "aubio.cvec",                 /* tp_name           */
  sizeof (Py_cvec),             /* tp_basicsize      */
  0,                            /* tp_itemsize       */
  (destructor) Py_cvec_del,     /* tp_dealloc        */
  0,                            /* tp_print          */
  0,                            /* tp_getattr        */
  0,                            /* tp_setattr        */
  0,                            /* tp_compare        */
  (reprfunc) Py_cvec_repr,      /* tp_repr           */
  0,                            /* tp_as_number      */
  0, //&Py_cvec_tp_as_sequence,      /* tp_as_sequence    */
  0,                            /* tp_as_mapping     */
  0,                            /* tp_hash           */
  0,                            /* tp_call           */
  0,                            /* tp_str            */
  0,                            /* tp_getattro       */
  0,                            /* tp_setattro       */
  0,                            /* tp_as_buffer      */
  Py_TPFLAGS_DEFAULT,           /* tp_flags          */
  Py_cvec_doc,                  /* tp_doc            */
  0,                            /* tp_traverse       */
  0,                            /* tp_clear          */
  0,                            /* tp_richcompare    */
  0,                            /* tp_weaklistoffset */
  0,                            /* tp_iter           */
  0,                            /* tp_iternext       */
  Py_cvec_methods,              /* tp_methods        */
  Py_cvec_members,              /* tp_members        */
  Py_cvec_getseters,            /* tp_getset         */
  0,                            /* tp_base           */
  0,                            /* tp_dict           */
  0,                            /* tp_descr_get      */
  0,                            /* tp_descr_set      */
  0,                            /* tp_dictoffset     */
  (initproc) Py_cvec_init,      /* tp_init           */
  0,                            /* tp_alloc          */
  Py_cvec_new,                  /* tp_new            */
};

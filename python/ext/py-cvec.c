#include "aubio-types.h"

/* cvec type definition

class cvec():
    def __new__(self, length = 1024):
        self.length = length / 2 + 1
        self.norm = np.zeros(length / 2 + 1)
        self.phas = np.zeros(length / 2 + 1)

*/

// special python type for cvec
typedef struct
{
  PyObject_HEAD
  PyObject *norm;
  PyObject *phas;
  uint_t length;
} Py_cvec;

static char Py_cvec_doc[] = "cvec object";

PyObject *
PyAubio_CCvecToPyCvec (cvec_t * input) {
  if (input == NULL) {
      PyErr_SetString (PyExc_ValueError, "PyAubio_CCvecToPyCvec got a null cvec!");
      return NULL;
  }
  Py_cvec* vec = (Py_cvec*) PyObject_New (Py_cvec, &Py_cvecType);
  npy_intp dims[] = { input->length, 1 };
  vec->norm = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, input->norm);
  vec->phas = PyArray_SimpleNewFromData (1, dims, AUBIO_NPY_SMPL, input->phas);
  vec->length = input->length;
  return (PyObject *)vec;
}

int
PyAubio_PyCvecToCCvec (PyObject *input, cvec_t *i) {
  if (PyObject_TypeCheck (input, &Py_cvecType)) {
      Py_cvec * in = (Py_cvec *)input;
      if (in->norm == NULL) {
        npy_intp dims[] = { in->length, 1 };
        in->norm = PyArray_ZEROS(1, dims, AUBIO_NPY_SMPL, 0);
      }
      if (in->phas == NULL) {
        npy_intp dims[] = { in->length, 1 };
        in->phas = PyArray_ZEROS(1, dims, AUBIO_NPY_SMPL, 0);
      }
      i->norm = (smpl_t *) PyArray_GETPTR1 ((PyArrayObject *)(in->norm), 0);
      i->phas = (smpl_t *) PyArray_GETPTR1 ((PyArrayObject *)(in->phas), 0);
      i->length = ((Py_cvec*)input)->length;
      return 1;
  } else {
      PyErr_SetString (PyExc_ValueError, "input array should be aubio.cvec");
      return 0;
  }
}

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
  self->norm = NULL;
  self->phas = NULL;
  return 0;
}

static void
Py_cvec_del (Py_cvec * self)
{
  Py_XDECREF(self->norm);
  Py_XDECREF(self->phas);
  Py_TYPE(self)->tp_free ((PyObject *) self);
}

static PyObject *
Py_cvec_repr (Py_cvec * self, PyObject * unused)
{
  PyObject *format = NULL;
  PyObject *args = NULL;
  PyObject *result = NULL;

  format = PyUnicode_FromString ("aubio cvec of %d elements");
  if (format == NULL) {
    goto fail;
  }

  args = Py_BuildValue ("I", self->length);
  if (args == NULL) {
    goto fail;
  }
  // hide actual norm / phas content

  result = PyUnicode_Format (format, args);

fail:
  Py_XDECREF (format);
  Py_XDECREF (args);

  return result;
}

PyObject *
Py_cvec_get_norm (Py_cvec * self, void *closure)
{
  // if it norm hasn't been created, create it now
  if (self->norm == NULL) {
    npy_intp dims[] = { self->length, 1 };
    self->norm = PyArray_ZEROS(1, dims, AUBIO_NPY_SMPL, 0);
  }
  Py_INCREF(self->norm);
  return (PyObject*)(self->norm);
}

PyObject *
Py_cvec_get_phas (Py_cvec * self, void *closure)
{
  // if it phas hasn't been created, create it now
  if (self->phas == NULL) {
    npy_intp dims[] = { self->length, 1 };
    self->phas = PyArray_ZEROS(1, dims, AUBIO_NPY_SMPL, 0);
  }
  Py_INCREF(self->phas);
  return (PyObject *)(self->phas);
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
      if (vec->length != PyArray_SIZE (array)) {
          PyErr_Format (PyExc_ValueError,
                  "input array has length %d, but cvec has length %d",
                  (int)PyArray_SIZE (array), vec->length);
          goto fail;
      }
    }

    Py_XDECREF(vec->norm);
    vec->norm = input;
    Py_INCREF(vec->norm);

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array as input");
    return 1;
  }

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
      if (vec->length != PyArray_SIZE (array)) {
          PyErr_Format (PyExc_ValueError,
                  "input array has length %d, but cvec has length %d",
                  (int)PyArray_SIZE (array), vec->length);
          goto fail;
      }
    }

    Py_XDECREF(vec->phas);
    vec->phas = input;
    Py_INCREF(vec->phas);

  } else {
    PyErr_SetString (PyExc_ValueError, "can only accept array as input");
    return 1;
  }

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
  PyVarObject_HEAD_INIT(NULL, 0)
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
  0, //&Py_cvec_tp_as_sequence, /* tp_as_sequence    */
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

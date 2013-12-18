#include "aubiowraphell.h"

static char Py_fft_doc[] = "fft object";

AUBIO_DECLARE(fft, uint_t win_s)

//AUBIO_NEW(fft)
static PyObject *
Py_fft_new (PyTypeObject * type, PyObject * args, PyObject * kwds)
{
  int win_s = 0;
  Py_fft *self;
  static char *kwlist[] = { "win_s", NULL };

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|I", kwlist,
          &win_s)) {
    return NULL;
  }

  self = (Py_fft *) type->tp_alloc (type, 0);

  if (self == NULL) {
    return NULL;
  }

  self->win_s = Py_default_vector_length;

  if (win_s > 0) {
    self->win_s = win_s;
  } else if (win_s < 0) {
    PyErr_SetString (PyExc_ValueError,
        "can not use negative window size");
    return NULL;
  }

  return (PyObject *) self;
}


AUBIO_INIT(fft, self->win_s)

AUBIO_DEL(fft)

static PyObject * 
Py_fft_do(PyObject * self, PyObject * args)
{
  PyObject *input;
  fvec_t *vec;
  cvec_t *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCFvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = new_cvec(((Py_fft *) self)->win_s);

  // compute the function
  aubio_fft_do (((Py_fft *)self)->o, vec, output);
  return (PyObject *)PyAubio_CCvecToPyCvec(output);
}

AUBIO_MEMBERS_START(fft) 
  {"win_s", T_INT, offsetof (Py_fft, win_s), READONLY,
    "size of the window"},
AUBIO_MEMBERS_STOP(fft)

static PyObject * 
Py_fft_rdo(Py_fft * self, PyObject * args)
{
  PyObject *input;
  cvec_t *vec;
  fvec_t *output;

  if (!PyArg_ParseTuple (args, "O", &input)) {
    return NULL;
  }

  vec = PyAubio_ArrayToCCvec (input);

  if (vec == NULL) {
    return NULL;
  }

  output = new_fvec(self->win_s);

  // compute the function
  aubio_fft_rdo (((Py_fft *)self)->o, vec, output);
  return (PyObject *)PyAubio_CFvecToArray(output);
}

static PyMethodDef Py_fft_methods[] = {
  {"rdo", (PyCFunction) Py_fft_rdo, METH_VARARGS,
    "synthesis of spectral grain"},
  {NULL}
};

AUBIO_TYPEOBJECT(fft, "aubio.fft")
